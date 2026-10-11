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

"""Rebuild checked reference states using literal defaults, preserving raw evidence."""

import argparse
import hashlib
import json
from pathlib import Path

from agent_system.environments.env_package.ftrl.core import acquired_atoms, score_trace
from agent_system.environments.env_package.ftrl.state_defaults import STATE_MODE, extract_defaults


def prepare(root, output):
    output.mkdir(parents=True, exist_ok=True)
    tasks, defaults, skipped = {}, {}, []
    for split in ("train", "test"):
        for line in (root / (split + ".jsonl")).open():
            task = json.loads(line)
            key = task["task_id"]
            assert key not in tasks
            tasks[key] = task
            defaults[key], omissions = extract_defaults(task["codes"])
            skipped.extend({"task_id": key, **x} for x in omissions)
    asset = output / "state_defaults_v1.json"
    asset.write_text(json.dumps({"state_mode": STATE_MODE, "tasks": defaults}, ensure_ascii=False, sort_keys=True))
    sha = hashlib.sha256(asset.read_bytes()).hexdigest()
    refs, extra_excluded, changed = [], [], 0
    for line in (root / "references_state_checked.jsonl").open():
        ref = json.loads(line)
        key = ref["task_id"]
        events, unique = [], []
        old_final = ref["final_atoms"]
        for step in ref["steps"]:
            step["before"] = acquired_atoms(events, defaults[key])
            event = {"calls": [step["call"]], "outputs": [step["output"]]}
            events.append(event)
            step["after"] = acquired_atoms(events, defaults[key])
            if set(step["after"]) - set(step["before"]):
                unique.append(event)
        ref["final_atoms"] = acquired_atoms(events, defaults[key])
        changed += old_final != ref["final_atoms"]
        ref["unique_information_scores"] = score_trace(tasks[key], unique)
        if ref["valid"] and ref["unique_information_scores"]["solve_rate"] != 1:
            ref["valid"] = False
            ref["invalid_reason"] = "normalized_verifier_requires_repeated_information"
            extra_excluded.append(key)
        ref["state_mode"] = STATE_MODE
        ref["state_defaults_sha256"] = sha
        refs.append(ref)
    target = output / "references_normalized_v1.jsonl"
    target.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in refs))
    report = {
        "state_mode": STATE_MODE,
        "tasks": len(tasks),
        "references": len(refs),
        "valid_references": sum(r["valid"] for r in refs),
        "extra_excluded": extra_excluded,
        "changed_final_states": changed,
        "skipped_defaults": skipped,
        "tool_signatures": sum(len(x) for x in defaults.values()),
        "input_sha256": {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in ("train.jsonl", "test.jsonl", "references_state_checked.jsonl")},
        "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (asset, target)},
    }
    (output / "normalized_state_manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "skipped_defaults"}, indent=2))
    print("Skipped defaults/signatures:", len(skipped))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.output)
