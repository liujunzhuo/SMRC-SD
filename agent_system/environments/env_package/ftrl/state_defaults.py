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

"""Literal function defaults for query-bound state equivalence; no code execution."""

import ast
import hashlib
import json
from pathlib import Path

STATE_MODE = "bound_defaults_v1"


def _json_literal(value):
    if value is None or type(value) in (str, bool, int, float):
        return True
    if isinstance(value, list):
        return all(_json_literal(x) for x in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_literal(v) for k, v in value.items())
    return False


def extract_defaults(codes):
    defaults, skipped = {}, []
    for name, code in codes.items():
        defaults[name] = {}
        try:
            # The executor resolves the last top-level definition of the tool.
            fn = [x for x in ast.parse(code).body if isinstance(x, ast.FunctionDef) and x.name == name][-1]
        except (SyntaxError, IndexError):
            skipped.append({"tool": name, "reason": "unavailable_signature"})
            continue
        args = fn.args.posonlyargs + fn.args.args
        pairs = list(zip(args[len(args) - len(fn.args.defaults) :], fn.args.defaults))
        pairs += [(a, d) for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults) if d is not None]
        for arg, node in pairs:
            try:
                value = ast.literal_eval(node)
                if not _json_literal(value):
                    raise ValueError("Not a JSON literal")
                json.dumps(value, allow_nan=False)
            except (ValueError, TypeError, SyntaxError):
                skipped.append({"tool": name, "argument": arg.arg, "reason": "nonliteral_or_nonjson_default"})
                continue
            defaults[name][arg.arg] = value
    return defaults, skipped


def load_state_defaults(path):
    if not path:
        return None
    raw = Path(path).read_bytes()
    artifact = json.loads(raw)
    if artifact["state_mode"] != STATE_MODE or not isinstance(artifact["tasks"], dict):
        raise ValueError("Invalid FTRL state-default artifact")
    artifact["sha256"] = hashlib.sha256(raw).hexdigest()
    return artifact


def task_defaults(artifact, task_id):
    # Missing tasks must fail closed instead of mixing normalized/raw atoms.
    return artifact["tasks"][task_id] if artifact is not None else None
