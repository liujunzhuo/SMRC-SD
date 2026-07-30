# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Score the fixed SMRC-SD anchor suite with a baseline or GRPO actor checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from probe_path_teacher_signal import FSDPCausalScorer, HFCausalScorer
from verl.trainer.ppo.path_privileged_utils import insert_privileged_context


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_dict(score: Any) -> dict[str, Any]:
    return {
        "mean_logprob": score.mean_logprob,
        "action_mean_logprob": score.action_mean_logprob,
        "response_len": score.response_len,
    }


def valid(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def contexts_for(anchor: dict[str, Any]) -> list[str]:
    if anchor["category"] == "B":
        return ["none", "skill", "other_success_same_task_type", "shuffled_path", "full_path", "smrc"]
    if anchor["category"] == "A":
        return ["none", "full_path", "smrc"]
    return ["none", "full_path"]


def build_pairs(anchors: list[dict[str, Any]]) -> tuple[list[tuple[str, str]], list[tuple[int, str, str]]]:
    pairs, keys = [], []
    for index, anchor in enumerate(anchors):
        for context_name in contexts_for(anchor):
            context = "" if context_name == "none" else str(anchor["contexts"][context_name])
            prompt = insert_privileged_context(str(anchor["input"]), context)
            pairs.append((prompt, str(anchor["output"])))
            keys.append((index, context_name, "sampled"))
            if anchor["category"] == "B":
                pairs.append((prompt, str(anchor["candidate_response"])))
                keys.append((index, context_name, "candidate"))
    return pairs, keys


def length_bucket_order(pairs: list[tuple[str, str]], scorer: Any, *, enabled: bool) -> list[int]:
    """Return a stable scoring order while retaining original pair identities."""
    if not enabled:
        return list(range(len(pairs)))
    lengths = []
    for pair_index, (prompt, response) in enumerate(pairs):
        prompt_ids, response_ids = scorer._encode_pair(prompt, response)
        lengths.append((len(prompt_ids) + len(response_ids), pair_index))
    return [pair_index for _length, pair_index in sorted(lengths)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    margins: dict[tuple[str, str], list[float]] = defaultdict(list)
    gaps: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        split = str(row["anchor_split"])
        scores = row["scores"]
        if row["category"] == "B":
            for context, values in scores.items():
                sampled, candidate = valid(values.get("sampled", {}).get("action_mean_logprob")), valid(values.get("candidate", {}).get("action_mean_logprob"))
                if sampled is not None and candidate is not None:
                    margins[(split, context)].append(candidate - sampled)
        else:
            baseline = valid(scores.get("none", {}).get("sampled", {}).get("action_mean_logprob"))
            for context in ("full_path", "smrc"):
                current = valid(scores.get(context, {}).get("sampled", {}).get("action_mean_logprob"))
                if baseline is not None and current is not None:
                    gaps[(split, context)].append(current - baseline)
    return {
        "n_anchors": len(rows),
        "B_action_margin": {
            f"{split}/{context}": {"n": len(values), "mean": mean(values), "positive_rate": mean([float(value > 0) for value in values])}
            for (split, context), values in sorted(margins.items())
        },
        "A_C_sampled_action_gap_vs_none": {
            f"{split}/{context}": {"n": len(values), "mean": mean(values), "positive_rate": mean([float(value > 0) for value in values])}
            for (split, context), values in sorted(gaps.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=["fsdp", "hf"], default="fsdp")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--disable-length-bucketing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anchors = read_jsonl(Path(args.anchor_manifest))
    pairs, keys = build_pairs(anchors)
    if args.backend == "fsdp":
        scorer = FSDPCausalScorer(args.model_path, dtype=args.dtype, trust_remote_code=False, max_model_len=args.max_model_len)
        rank, world_size = scorer.rank, scorer.world_size
        ordered_to_original = length_bucket_order(pairs, scorer, enabled=not args.disable_length_bucketing)
        scoring_pairs = [pairs[pair_index] for pair_index in ordered_to_original]
        if len(scoring_pairs) % (world_size * args.batch_size) != 0:
            raise ValueError(
                "FSDP scoring requires an equal number of full batches on every rank. "
                f"Got {len(scoring_pairs)} pairs for world_size={world_size}, batch_size={args.batch_size}."
            )
        # FSDPCausalScorer already shards its input by rank. Passing a pre-sharded
        # list here would shard it a second time and leave most pair scores absent.
        local_scores = scorer.score_pairs(scoring_pairs, args.batch_size)
        local_positions = list(range(rank, len(scoring_pairs), world_size))
        local_payload = [(ordered_to_original[position], score_dict(score)) for position, score in zip(local_positions, local_scores)]
        gathered: list[Any] = [None] * world_size
        torch.distributed.all_gather_object(gathered, local_payload)
        if rank != 0:
            torch.distributed.barrier()
            return
        merged = {index: payload for shard in gathered for index, payload in shard}
    else:
        scorer = HFCausalScorer(args.model_path, dtype=args.dtype, device=args.device, trust_remote_code=False, max_model_len=args.max_model_len)
        ordered_to_original = length_bucket_order(pairs, scorer, enabled=not args.disable_length_bucketing)
        scoring_pairs = [pairs[pair_index] for pair_index in ordered_to_original]
        merged = {
            original_index: score_dict(score)
            for original_index, score in zip(ordered_to_original, scorer.score_pairs(scoring_pairs, args.batch_size))
        }

    output_rows = []
    for index, anchor in enumerate(anchors):
        scores: dict[str, dict[str, Any]] = defaultdict(dict)
        for pair_index, (anchor_index, context, response_kind) in enumerate(keys):
            if anchor_index == index:
                scores[context][response_kind] = merged[pair_index]
        output_rows.append({key: value for key, value in anchor.items() if key not in {"input", "output", "candidate_response", "contexts"}} | {"scores": dict(scores)})
    summary = summarize(output_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# SMRC-SD Anchor Probe", "", f"- Anchors: `{summary['n_anchors']}`", "", "## B: candidate minus sampled action margin", ""]
    for key, value in summary["B_action_margin"].items():
        lines.append(f"- `{key}`: n={value['n']}, mean={value['mean']:.6f}, positive={value['positive_rate']:.3f}")
    lines.extend(["", "## A/C: sampled action gap versus no privilege", ""])
    for key, value in summary["A_C_sampled_action_gap_vs_none"].items():
        lines.append(f"- `{key}`: n={value['n']}, mean={value['mean']:.6f}, positive={value['positive_rate']:.3f}")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir}")
    if args.backend == "fsdp":
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
