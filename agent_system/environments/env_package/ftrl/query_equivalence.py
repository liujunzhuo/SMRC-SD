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

"""Conservative, versioned query equivalence; never execute tool source."""

import ast
import hashlib
import json
from pathlib import Path

QUERY_MODE = "literal_lower_v1"


def fields_for_code(code, name):
    try:
        tree = ast.parse(code)
        fn = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name][-1]
    except (SyntaxError, IndexError):
        return []
    if any(isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)) and x is not fn for x in ast.walk(fn)):
        return []
    if any(isinstance(x, ast.Name) and x.id in {"locals", "globals", "eval", "exec", "vars", "getattr"} for x in ast.walk(fn)):
        return []
    parents = {child: parent for parent in ast.walk(fn) for child in ast.iter_child_nodes(parent)}
    fields = []
    for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
        uses = [x for x in ast.walk(fn) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and x.id == arg.arg]
        lowered = [x for x in uses if isinstance(parents[x], ast.Attribute) and parents[x].attr == "lower" and isinstance(parents.get(parents[x]), ast.Call) and not parents[parents[x]].args and not parents[parents[x]].keywords]
        if not lowered or any(isinstance(x, ast.Name) and isinstance(x.ctx, (ast.Store, ast.Del)) and x.id == arg.arg for x in ast.walk(fn)):
            continue

        def allowed(node, lowered=lowered):
            parent = parents[node]
            if node in lowered:
                return True
            if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id == "isinstance" and len(parent.args) == 2 and parent.args[0] is node and isinstance(parent.args[1], ast.Name) and parent.args[1].id == "str":
                return True
            if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
                return True
            # Boolean operations are allowed only when their result is a truth test.
            while isinstance(parent, ast.BoolOp):
                node, parent = parent, parents.get(parent)
            return isinstance(parent, ast.If) and parent.test is node

        if all(allowed(node) for node in uses):
            fields.append(arg.arg)
    return fields


def load_query_equivalence(path, defaults):
    if not path:
        return None
    raw = Path(path).read_bytes()
    artifact = json.loads(raw)
    if artifact["mode"] != QUERY_MODE or not isinstance(artifact["tasks"], dict):
        raise ValueError("Invalid query-equivalence artifact")
    if defaults is None or artifact["state_defaults_sha256"] != defaults["sha256"]:
        raise ValueError("Query-equivalence/default artifact hash mismatch")
    artifact["sha256"] = hashlib.sha256(raw).hexdigest()
    return artifact


def task_query_fields(artifact, task_id):
    return artifact["tasks"][task_id] if artifact is not None else None
