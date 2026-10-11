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

"""Convert pinned FTRL assets and replay references without model generation."""

import argparse
import ast
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--url", default="http://127.0.0.1:28980")
    args = parser.parse_args()
    root = Path(args.root)
    if not args.replay:
        import pyarrow as pa
        import pyarrow.parquet as pq

        audit = {}
        for split in ("train", "test"):
            source = root / "upstream" / f"{split}.parquet"
            rows = pq.read_table(source).to_pylist()
            target, dataset, imports, syntax_errors = [], [], Counter(), []
            for i, row in enumerate(rows):
                item = {"task_id": f"{split}:{i:04d}", "data_source": row["data_source"], "messages": row["messages"]}
                for key in ("tools", "codes", "unsolved_set", "ground_truth"):
                    value = row.get(key, [] if key == "ground_truth" else {})
                    item[key] = json.loads(value) if isinstance(value, str) else value
                item["answer"] = str(row["answer"])
                for name, code in item["codes"].items():
                    try:
                        tree = ast.parse(code)
                    except SyntaxError:
                        syntax_errors.append([item["task_id"], name])
                        continue
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            imports.update(a.name for a in node.names)
                        if isinstance(node, ast.ImportFrom):
                            imports[node.module] += 1
                target.append(item)
                dataset.append({"prompt": row["messages"], "data_source": "ftrl/" + row["data_source"], "env_kwargs": {"task_id": item["task_id"]}})
            with (root / f"{split}.jsonl").open("w") as f:
                for item in target:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            pq.write_table(pa.Table.from_pylist(dataset), root / f"{split}_rl.parquet")
            audit[split] = {"count": len(rows), "types": dict(Counter(x["data_source"] for x in rows)), "imports": dict(imports), "syntax_errors": syntax_errors, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        (root / "data_audit.json").write_text(json.dumps(audit, indent=2))
        print(json.dumps(audit, indent=2))
        return
    from agent_system.environments.env_package.ftrl.client import Client
    from agent_system.environments.env_package.ftrl.core import acquired_atoms, render_call, score_trace

    tasks = [json.loads(line) for line in (root / "train.jsonl").open()]

    def replay(task):
        client = Client(args.url)
        calls = [{"name": x["name"], "arguments": x["parameters"]} for x in task["ground_truth"]]
        outputs = client.request("execute", task_id=task["task_id"], calls=calls)
        events, steps = [], []
        for call, output in zip(calls, outputs, strict=True):
            before = acquired_atoms(events)
            events.append({"calls": [call], "outputs": [output]})
            steps.append({"call": call, "output": output, "before": before, "after": acquired_atoms(events)})
        scores = score_trace(task, events)
        final = "<answer>" + task["answer"] + "</answer>"
        valid = scores["solve_rate"] == 1 and bool(acquired_atoms(events))
        return {
            "task_id": task["task_id"],
            "data_source": task["data_source"],
            "valid": valid,
            "scores": scores,
            "steps": steps,
            "final_atoms": acquired_atoms(events),
            "final_response": final,
            "full_path": "\n\n".join(render_call(s["call"]) + "\n<tool_response>\n" + s["output"] + "\n</tool_response>" for s in steps) + "\n\n" + final,
        }

    with ThreadPoolExecutor(max_workers=16) as pool:
        references = list(pool.map(replay, tasks))
    with (root / "references.jsonl").open("w") as f:
        for reference in references:
            f.write(json.dumps(reference, ensure_ascii=False) + "\n")
    summary = {typ: {"total": sum(x["data_source"] == typ for x in references), "valid": sum(x["data_source"] == typ and x["valid"] for x in references)} for typ in sorted({x["data_source"] for x in references})}
    summary["invalid_ids"] = [x["task_id"] for x in references if not x["valid"]]
    (root / "reference_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
