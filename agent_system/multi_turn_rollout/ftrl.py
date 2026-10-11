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

"""FTRL native tool-call rollouts with raw-token history and per-turn loss rows."""

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from agent_system.environments.env_package.ftrl.client import Client
from agent_system.environments.env_package.ftrl.core import acquired_atoms, parse_calls
from agent_system.environments.env_package.ftrl.prompts import student_messages, tool_feedback
from agent_system.environments.env_package.ftrl.query_equivalence import load_query_equivalence, task_query_fields
from agent_system.environments.env_package.ftrl.state_defaults import load_state_defaults, task_defaults
from agent_system.multi_turn_rollout.active_batch import generate_balanced_active_batch
from agent_system.multi_turn_rollout.utils import to_list_of_dict
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask


def prompt_row(ids, tokenizer, width):
    if len(ids) > width:
        raise ValueError("FTRL prompt must not be silently truncated")
    pad = width - len(ids)
    tokens = torch.tensor([tokenizer.pad_token_id] * pad + ids, dtype=torch.long)
    mask = torch.tensor([0] * pad + [1] * len(ids), dtype=torch.long)
    return {"input_ids": tokens, "attention_mask": mask, "position_ids": compute_position_id_with_mask(mask[None])[0], "raw_prompt_ids": ids}


class FTRLCollector:
    def __init__(self, config, tokenizer, processor=None):
        self.config, self.tokenizer = config, tokenizer
        self.client = Client(config.env.ftrl.url)
        self.tasks = {}
        self.state_defaults = load_state_defaults(config.env.ftrl.get("state_defaults"))
        self.query_equivalence = load_query_equivalence(config.env.ftrl.get("query_equivalence"), self.state_defaults)

    def task(self, task_id):
        if task_id not in self.tasks:
            self.tasks[task_id] = self.client.request("task", task_id=task_id)
        return self.tasks[task_id]

    def multi_turn_loop(self, gen_batch, actor_rollout_wg, envs=None, is_train=True, global_step=0):
        repeat = self.config.env.rollout.n if is_train else 1
        inputs = gen_batch.repeat(repeat_times=repeat, interleave=True)
        task_ids = [x["task_id"] for x in inputs.non_tensor_batch["env_kwargs"]]
        tasks = [self.task(key) for key in task_ids]
        prompt_style = self.config.env.ftrl.get("prompt_style", "upstream")
        prefix = [self.tokenizer.apply_chat_template(student_messages(x["messages"], prompt_style), tools=x["tools"], tokenize=True, add_generation_prompt=True, enable_thinking=self.config.env.ftrl.enable_thinking) for x in tasks]
        n = len(tasks)
        events, rows = [[] for _ in tasks], [[] for _ in tasks]
        done = [False] * n
        end_reason = ["max_turns"] * n
        uid = [str(uuid.uuid4()) for _ in range(len(gen_batch))]
        uid = [x for x in uid for _ in range(repeat)]
        traj = [str(uuid.uuid4()) for _ in tasks]
        metrics = {"active_requests": 0, "padding_requests": 0, "generation_wall_seconds": 0.0, "environment_wall_seconds": 0.0}
        width = self.config.data.max_prompt_length
        keep_awake = self.config.actor_rollout_ref.rollout.keep_engine_awake_during_multiturn
        if keep_awake:
            actor_rollout_wg.begin_rollout_generation()
        try:
            with ThreadPoolExecutor(max_workers=16) as pool:
                for turn in range(self.config.env.max_steps):
                    for i in range(n):
                        if not done[i] and len(prefix[i]) > width:
                            if not rows[i]:
                                raise ValueError(f"Initial FTRL prompt exceeds budget: {task_ids[i]} {len(prefix[i])}")
                            done[i], end_reason[i] = True, "context_limit"
                    active = [i for i in range(n) if not done[i]]
                    if not active:
                        break
                    batch = DataProto.from_single_dict(collate_fn([prompt_row(prefix[i], self.tokenizer, width) for i in active]), meta_info=dict(gen_batch.meta_info))
                    started = time.perf_counter()
                    output, pad = generate_balanced_active_batch(batch, actor_rollout_wg)
                    metrics["generation_wall_seconds"] += time.perf_counter() - started
                    metrics["active_requests"] += len(active)
                    metrics["padding_requests"] += pad
                    generated = to_list_of_dict(output)
                    pending = []
                    for k, i in enumerate(active):
                        row = generated[k]
                        response_mask = row["attention_mask"][-row["responses"].shape[-1] :].bool()
                        response_ids = row["responses"][response_mask].tolist()
                        response = self.tokenizer.decode(response_ids, skip_special_tokens=False)
                        calls = parse_calls(response)
                        if self.config.env.ftrl.get("audit_dir"):
                            audit_root = Path(self.config.env.ftrl.audit_dir)
                            audit_root.mkdir(parents=True, exist_ok=True)
                            with (audit_root / f"requests_{'train' if is_train else 'val'}_{global_step}.jsonl").open("a") as handle:
                                handle.write(json.dumps({"task_id": task_ids[i], "traj_uid": traj[i], "turn": turn, "response": response, "calls": calls}, ensure_ascii=False) + "\n")
                        row.update(
                            {
                                "uid": uid[i],
                                "traj_uid": traj[i],
                                "task_key": task_ids[i],
                                "gamefile": "ftrl/" + task_ids[i],
                                "data_source": "ftrl/" + tasks[i]["data_source"],
                                "turn_step": turn,
                                "ftrl_atoms": acquired_atoms(events[i], task_defaults(self.state_defaults, task_ids[i]), task_query_fields(self.query_equivalence, task_ids[i])),
                                "ftrl_history": list(events[i]),
                                "executed_action": json.dumps(calls, ensure_ascii=False),
                                "is_action_valid": bool(calls) or "<answer>" in response,
                                "active_masks": True,
                                "ftrl_response_truncated": len(response_ids) == self.config.data.max_response_length and response_ids[-1] != self.tokenizer.eos_token_id,
                            }
                        )
                        for name in ("validation_item_id", "validation_sample_index"):
                            if name in inputs.non_tensor_batch:
                                row[name] = inputs.non_tensor_batch[name][i]
                        rows[i].append(row)
                        prefix[i].extend(response_ids)
                        if not calls:
                            done[i], end_reason[i] = True, "no_tool_call"
                            continue
                        pending.append((i, calls, pool.submit(self.client.request, "execute", task_id=task_ids[i], calls=calls)))
                    started = time.perf_counter()
                    for i, calls, future in pending:
                        outputs = future.result()
                        events[i].append({"calls": calls, "outputs": outputs})
                        # Same user/tool_response wrapper as TurnSight; preserve sampled token IDs.
                        feedback = tool_feedback(outputs, prefix[i][-1] == self.tokenizer.eos_token_id, self.config.env.ftrl.enable_thinking)
                        prefix[i].extend(self.tokenizer.encode(feedback, add_special_tokens=False))
                    metrics["environment_wall_seconds"] += time.perf_counter() - started
                    print(f"[FTRL] step={global_step} train={is_train} turn={turn} active={len(active)} calls={sum(len(c) for _, c, _ in pending)} finished={sum(done)}", flush=True)
                scores = list(pool.map(lambda i: self.client.request("score", task_id=task_ids[i], events=events[i]), range(n)))
        finally:
            if keep_awake:
                actor_rollout_wg.end_rollout_generation()
        effective = []
        for i in range(n):
            for row in rows[i]:
                row.update({"episode_rewards": scores[i]["solve_f1"], "episode_lengths": len(rows[i]), "tool_callings": scores[i]["calls"], "ftrl_scores": scores[i], "ftrl_end_reason": end_reason[i]})
                effective.append(row)
        metrics.update({"episodes": n, "context_limit_ratio": end_reason.count("context_limit") / n, "response_truncated_ratio": sum(r["ftrl_response_truncated"] for r in effective) / len(effective)})
        result = DataProto.from_single_dict(collate_fn(effective))
        # collate_fn uses object arrays for metadata; framework reductions expect
        # numeric NumPy scalars for episode statistics.
        for key, dtype in (("episode_rewards", np.float64), ("episode_lengths", np.int64), ("tool_callings", np.int64)):
            result.non_tensor_batch[key] = result.non_tensor_batch[key].astype(dtype)
        result.meta_info["rollout_metrics"] = {"ftrl/rollout/" + k: v for k, v in metrics.items()}
        audit_root = self.config.env.ftrl.get("audit_dir")
        if audit_root:
            root = Path(audit_root)
            root.mkdir(parents=True, exist_ok=True)
            with (root / f"{'train' if is_train else 'val'}_{global_step}_{uuid.uuid4().hex[:8]}.jsonl").open("w") as handle:
                for i in range(n):
                    handle.write(json.dumps({"task_id": task_ids[i], "traj_uid": traj[i], "events": events[i], "scores": scores[i], "end_reason": end_reason[i]}, ensure_ascii=False) + "\n")
        return result
