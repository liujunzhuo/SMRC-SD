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

"""Disposable CPU child. No model packages are imported in the tool environment."""

import ast
import contextlib
import io
import json
import resource
import sys

# Keep exec inside a function, as in TurnSight's call_function. Defining tool
# code directly in its globals silently fixes upstream global-name errors.
_CALL_FUNCTION = """def call_function(name, arguments, code):
    exec(code)
    return eval(name)(**arguments)
"""


def execute(codes, calls):
    outputs = []
    for call in calls:
        name = call["name"]
        try:
            code = codes.get(name)
            if not code:
                raise ValueError(f"No code found for tool {name}")
            # This is an audited local benchmark interpreter, not a general code service.
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                    if any(
                        n.split(".")[0] not in {"json", "math", "re", "datetime", "collections", "statistics", "itertools", "typing", "dateutil", "pytz", "dicttoxml", "os", "csv", "io", "xml", "calendar", "cmath", "warnings", "fractions", "operator", "string", "numbers", "decimal"} for n in names
                    ):
                        raise ValueError("Unsupported benchmark import")
            namespace = {"json": json}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(_CALL_FUNCTION, namespace)
                result = namespace["call_function"](name, call["arguments"], code)
            outputs.append(json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result))
        except Exception as exc:
            outputs.append(f"an error occured when call {name}: {exc}")
    return outputs


if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 11))
    request = json.load(sys.stdin)
    print(json.dumps(execute(request["codes"], request["calls"]), ensure_ascii=False))
