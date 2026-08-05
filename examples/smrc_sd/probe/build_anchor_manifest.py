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

"""Build the fixed, quality-controlled SMRC-SD teacher-probe anchor manifest."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from probe_path_teacher_signal import TrajectoryLookup
from verl.trainer.ppo.path_privileged_utils import PathPrivilegedContextProvider, extract_admissible_actions, extract_current_observation
from verl.trainer.ppo.rlsd_utils import SkillProvider

ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
PATH_RE = re.compile(r"Complete successful path for this task:\s*\n([^\n]+)", re.IGNORECASE)
TASK_RE = re.compile(r"(pick_and_place|pick_clean_then_place_in_recep|pick_heat_then_place_in_recep|pick_cool_then_place_in_recep|look_at_obj_in_light|pick_two_obj_and_place)")


@dataclass
class ProbeBatch:
    non_tensor_batch: dict[str, np.ndarray]


def normalize(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def extract_action(text: str) -> str:
    match = ACTION_RE.search(text or "")
    return " ".join(match.group(1).strip().split()) if match else ""


def replace_action(response: str, action: str) -> str:
    if not action:
        return ""
    if ACTION_RE.search(response or ""):
        return ACTION_RE.sub(f"<action>{action}</action>", response, count=1)
    return f"<action>{action}</action>"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_batch(rows: list[dict[str, Any]]) -> ProbeBatch:
    keys = set().union(*(row.keys() for row in rows))
    return ProbeBatch(non_tensor_batch={key: np.array([row.get(key) for row in rows], dtype=object) for key in keys})


def config(*, teacher_context: str, trajectory_index: str) -> Any:
    is_smrc = teacher_context == "gt_recovery_aligned"
    return OmegaConf.create(
        {
            "algorithm": {
                "path_opd": {
                    "teacher_context": teacher_context,
                    "strict_gt_path": True,
                    "max_remaining_actions": 20,
                    "include_executed_prefix": True,
                    "include_remaining_path": True,
                    "include_full_path": True,
                    "path_source_policy": "canonical",
                    "path_buffer_path": "",
                    "filter_no_progress_success_actions": True,
                    "enable_history_suffix_alignment": False,
                    "alignment_mode": "dynamic",
                    "enable_walkthrough_trace_recovery": True,
                    "recovery_match_mode": "state",
                    "recovery_state_domain": "alfworld",
                    "recovery_alfworld_object_match_mode": "exact",
                    "state_sdl_selection": "match_only" if is_smrc else "all",
                    "state_match_context": "recovery",
                    "recovery_prompt_style": "state_summary_candidate" if is_smrc else "current",
                    "recovery_guidance_style": "path_on_or_off" if is_smrc else "style_default",
                    "recovery_use_last_progress_observation_on_no_progress": False,
                    "recovery_correction_only": False,
                    "recovery_success_confirmation_sdl_weight": 1.0,
                    "recovery_skip_if_action_matches_student": False,
                    "trajectory_index_path": trajectory_index,
                    "sdl_confidence_weighting": False,
                    "sdl_drop_repeated_action_after_turn": False,
                    "policy_loss_filter": "none",
                }
            }
        }
    )


def candidate_from_context(context: str) -> str:
    match = re.search(r"Candidate next action for the current state:\s*\n([^\n]+)", context or "", re.IGNORECASE)
    return match.group(1).strip() if match else ""


def shuffled_context(plain_context: str, *, seed: int) -> str:
    match = PATH_RE.search(plain_context or "")
    if not match:
        return plain_context
    actions = [part.strip() for part in match.group(1).split("->") if part.strip()]
    random.Random(seed).shuffle(actions)
    return path_context_with_actions(plain_context, actions)


def path_context_with_actions(plain_context: str, actions: list[str]) -> str:
    match = PATH_RE.search(plain_context or "")
    if not match or not actions:
        return plain_context
    return plain_context[: match.start(1)] + " -> ".join(actions) + plain_context[match.end(1) :]


def task_family(row: dict[str, Any]) -> str:
    match = TASK_RE.search(f"{row.get('gamefile') or ''} {row.get('input') or ''}")
    return match.group(1) if match else "unknown"


def quality_ok(row: dict[str, Any], progress: bool) -> bool:
    parsed, executed = normalize(extract_action(str(row.get("output") or ""))), normalize(row.get("executed_action"))
    admissible = {normalize(item) for item in extract_admissible_actions(str(row.get("input") or ""))}
    return bool(
        parsed
        and parsed == executed
        and executed in admissible
        and truthy(row.get("is_action_valid"))
        and not truthy(row.get("has_extra_output_after_action"))
        and progress
    )


def balanced_sample(rows: list[dict[str, Any]], quota: int, *, seed: int) -> list[dict[str, Any]]:
    if len(rows) < quota:
        raise ValueError(f"Need {quota} anchors but only {len(rows)} satisfy the filters.")
    buckets: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(int(row["source_step"]), str(row["task_family"]))].append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    chosen: list[dict[str, Any]] = []
    # Round-robin across source step and family; depleted buckets simply drop out.
    keys = sorted(buckets)
    while len(chosen) < quota:
        progressed = False
        for key in keys:
            if buckets[key] and len(chosen) < quota:
                chosen.append(buckets[key].pop())
                progressed = True
        if not progressed:
            break
    return chosen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-jsonl", nargs="+", required=True)
    parser.add_argument("--trajectory-index", default="examples/smrc_sd/data/alfworld_verified_walkthrough_prefixes.jsonl")
    parser.add_argument("--skills-dir", default="skills/alfworld")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--a-quota", type=int, default=400)
    parser.add_argument("--c-quota", type=int, default=400)
    parser.add_argument("--b-success-quota", type=int, default=200)
    parser.add_argument("--b-failure-quota", type=int, default=200)
    parser.add_argument("--max-observed-path-actions", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_counts: Counter[str] = Counter()
    lookup = TrajectoryLookup.from_jsonl(args.trajectory_index)
    for rollout_arg in args.rollout_jsonl:
        rollout_path = Path(rollout_arg)
        source_step = int(rollout_path.stem)
        rows = read_jsonl(rollout_path)
        rows.sort(key=lambda row: (str(row.get("traj_uid") or ""), int(row.get("turn_step") or 0)))
        by_traj: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_traj[str(row.get("traj_uid") or "")].append(row)
        terminal_success = {uid: any(float(row.get("score") or row.get("episode_rewards") or 0.0) > 0 for row in items) for uid, items in by_traj.items()}
        lengths = {uid: sum(bool(row.get("executed_action")) for row in items) for uid, items in by_traj.items()}
        progress: dict[int, bool] = {}
        for uid, items in by_traj.items():
            for index, row in enumerate(items):
                if index + 1 < len(items):
                    next_obs = extract_current_observation(str(items[index + 1].get("input") or ""))
                    progress[id(row)] = "nothing happens" not in " ".join(next_obs.lower().split())
                else:
                    progress[id(row)] = terminal_success[uid]

        smrc = PathPrivilegedContextProvider(config=config(teacher_context="gt_recovery_aligned", trajectory_index=args.trajectory_index))
        plain = PathPrivilegedContextProvider(config=config(teacher_context="gt_path", trajectory_index=args.trajectory_index))
        skill = PathPrivilegedContextProvider(config=config(teacher_context="skill", trajectory_index=args.trajectory_index), skill_provider=SkillProvider(args.skills_dir))
        for provider in (smrc, plain, skill):
            provider.start_batch_metrics()
        batch = make_batch(rows)
        for row_index, row in enumerate(rows):
            prompt = str(row.get("input") or "")
            smrc_context = smrc.build_context(batch=batch, idx=row_index, prompt_text=prompt)
            metrics = smrc._sample_metrics[-1]
            plain_context = plain.build_context(batch=batch, idx=row_index, prompt_text=prompt)
            skill_context = skill.build_context(batch=batch, idx=row_index, prompt_text=prompt)
            matched = float(metrics.get("path_opd/recovery_state_match_ratio", 0.0) or 0.0) > 0.5
            candidate = candidate_from_context(smrc_context)
            executed = str(row.get("executed_action") or "")
            correct_path, _ = lookup.get_correct_path(row)
            other_same_type = lookup.sample_wrong_same_task(
                row,
                correct_path,
                random.Random(args.seed + row_index + 1000003 * source_step),
            )
            if matched and candidate and normalize(candidate) == normalize(executed):
                category = "A"
            elif matched:
                category = "B"
            else:
                category = "C"
            raw_counts[category] += 1
            uid = str(row.get("traj_uid") or "")
            clean = quality_ok(row, progress[id(row)])
            if category in {"A", "C"}:
                eligible = clean and terminal_success[uid] and lengths[uid] <= args.max_observed_path_actions
                split = category
            else:
                eligible = clean and bool(candidate) and normalize(candidate) != normalize(executed) and bool(other_same_type)
                split = "B_success" if terminal_success[uid] else "B_failure"
            if not eligible:
                continue
            candidate_row = {
                "source_rollout_jsonl": str(rollout_path),
                "source_step": source_step,
                "source_row_index": row_index,
                "traj_uid": uid,
                "turn_step": int(row.get("turn_step") or 0),
                "gamefile": row.get("gamefile"),
                "task_family": task_family(row),
                "category": category,
                "anchor_split": split,
                "terminal_success": terminal_success[uid],
                "observed_path_actions": lengths[uid],
                "executed_action": executed,
                "candidate_action": candidate,
                "input": prompt,
                "output": str(row.get("output") or ""),
                "candidate_response": replace_action(str(row.get("output") or ""), candidate) if category == "B" else "",
                "contexts": {
                    "full_path": plain_context,
                    "smrc": smrc_context,
                    "skill": skill_context,
                    "shuffled_path": shuffled_context(plain_context, seed=args.seed + row_index + 1000003 * source_step),
                    "other_success_same_task_type": path_context_with_actions(plain_context, other_same_type),
                },
                "quality": {
                    "parsed_equals_executed": normalize(extract_action(str(row.get("output") or ""))) == normalize(executed),
                    "prompt_admissible": normalize(executed) in {normalize(item) for item in extract_admissible_actions(prompt)},
                    "env_valid": truthy(row.get("is_action_valid")),
                    "no_extra_output": not truthy(row.get("has_extra_output_after_action")),
                    "observed_progress": progress[id(row)],
                },
            }
            candidates[split].append(candidate_row)

    selected = []
    for split, quota, offset in (("A", args.a_quota, 11), ("C", args.c_quota, 23), ("B_success", args.b_success_quota, 37), ("B_failure", args.b_failure_quota, 53)):
        selected.extend(balanced_sample(candidates[split], quota, seed=args.seed + offset))
    selected.sort(key=lambda row: (row["anchor_split"], row["source_step"], row["task_family"], row["traj_uid"], row["turn_step"]))
    for index, row in enumerate(selected, start=1):
        row["anchor_id"] = f"SMRC-{row['anchor_split']}-{index:04d}-s{row['source_step']:03d}-{row['traj_uid'][:8]}-t{row['turn_step']:02d}"
    write_jsonl(Path(args.output), selected)
    by_split = Counter(row["anchor_split"] for row in selected)
    by_step = {split: dict(sorted(Counter(row["source_step"] for row in selected if row["anchor_split"] == split).items())) for split in sorted(by_split)}
    summary = {
        "anchors": len(selected),
        "by_split": dict(sorted(by_split.items())),
        "by_source_step": by_step,
        "raw_matcher_counts": dict(sorted(raw_counts.items())),
        "eligible_before_sampling": {key: len(value) for key, value in sorted(candidates.items())},
        "quality_contract": "parsed=executed; prompt-admissible; env-valid; no-extra-output; observed-progress; A/C terminal success and observed path length <= max",
        "max_observed_path_actions": args.max_observed_path_actions,
        "seed": args.seed,
    }
    summary_path = Path(args.summary) if args.summary else Path(args.output).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
