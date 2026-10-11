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

"""Pure-stdlib FTRL protocol; scoring follows the published execution verifier."""

import hashlib
import json
import re
from copy import deepcopy


def parse_calls(text):
    """TurnSight's tagged JSON interface, including multiple calls per turn."""
    start = text.find("<tool_call>")
    if start < 0:
        return []
    calls = []
    for section in text[start:].split("</tool_call>"):
        section = section.strip()
        if not section.startswith("<tool_call>"):
            continue
        try:
            obj = json.loads(section[len("<tool_call>") :].strip())
            args = obj["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            if obj.get("name") and isinstance(args, dict):
                calls.append({"name": obj["name"], "arguments": args})
        except (ValueError, KeyError, TypeError):
            continue
    return calls


def render_call(call):
    return "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>"


def answer_verify(predict, golden):
    # Upstream Code/utils/utils.py: comma-separated conjuncts, case-insensitive substring.
    if isinstance(predict, (dict, list)):
        predict = json.dumps(predict, ensure_ascii=False)
    predict = str(predict).lower().replace(",", "").strip()
    return all(item.lower().replace(",", "").strip() in predict for item in golden.split(", "))


def score_trace(task, events):
    pending = deepcopy(task["unsolved_set"])
    total = sum(map(len, pending.values()))
    calls = 0
    for event in events:
        for call, output in zip(event["calls"], event["outputs"], strict=True):
            calls += 1
            for answer in pending.get(call["name"], []):
                if answer_verify(output, answer):
                    pending[call["name"]].remove(answer)
                    break
    solved = total - sum(map(len, pending.values()))
    return {"solve_precision": solved / calls if calls else 1.0, "solve_rate": solved / total, "solve_f1": 2 * solved / (calls + total), "solved": solved, "required": total, "calls": calls}


def canonical_information(output):
    """Conservative response equivalence, not an LLM semantic judge or gold-answer test."""
    text = str(output).strip()
    if not text or re.match(r"(?i)^(an error|error\b|timeout\b|invalid arguments|no (?:data|results?|records?)\b|not found\b)", text):
        return None
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        value = text
    ignored = {"success", "error", "code", "status_code"}

    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in sorted(v.items()) if k.lower() not in ignored and clean(x) not in (None, "", [], {})}
        if isinstance(v, list):
            return [clean(x) for x in v if clean(x) not in (None, "", [], {})]
        if isinstance(v, str):
            v = re.sub(r"\s+", " ", v).strip().casefold()
            if v in {"none", "null", "unknown", "n/a", "ok", "success", "successful"} or re.match(r"^(?:error\b|not found\b|no\b.*\b(?:found|available)\b)", v):
                return None
            return v
        return v

    value = clean(value)
    if value in (None, "", [], {}) or isinstance(value, bool):
        return None
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def information_atom(output, call, defaults=None, query_lower_fields=None):
    value = canonical_information(output)
    # Bind the result to its query: two different entities returning "2" are
    # different facts. Cross-tool semantic equivalence is deliberately not guessed.
    query = dict(call)
    if defaults is not None:
        query["arguments"] = {**deepcopy(defaults.get(call["name"], {})), **call["arguments"]}
    if query_lower_fields:
        query["arguments"] = dict(query["arguments"])
        for field in query_lower_fields.get(call["name"], []):
            if isinstance(query["arguments"].get(field), str):
                query["arguments"][field] = query["arguments"][field].lower()
    grounded = json.dumps({"query": query, "result": value}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(grounded.encode()).hexdigest()[:24] if value is not None else None


def acquired_atoms(events, defaults=None, query_lower_fields=None):
    return sorted({atom for event in events for call, output in zip(event["calls"], event["outputs"], strict=True) if (atom := information_atom(output, call, defaults, query_lower_fields))})


def match_reference(reference, atoms, mode):
    """Match observed information only; never inspect the sampled next response/reward."""
    if mode not in {"exact", "subset"}:
        raise ValueError(mode)
    current = set(atoms)
    candidates = []
    for index, step in enumerate(reference.get("steps", [])):
        before, after = set(step["before"]), set(step["after"])
        # Do not suggest calls whose information has already been acquired.
        if after - before and (after - before) <= current:
            continue
        # An empty reference root is not a wildcard for arbitrary informative states.
        compatible = before == current if mode == "exact" else before <= current and (bool(before) or not current)
        if compatible:
            candidates.append((len(before), index, step))
    complete = set(reference.get("final_atoms", []))
    if complete and (complete == current if mode == "exact" else complete <= current):
        return {"matched": True, "index": len(reference["steps"]), "candidate": reference["final_response"], "kind": "completion", "anchor_atoms": sorted(complete)}
    if not candidates:
        return {"matched": False, "index": -1, "candidate": "", "kind": "unmatched", "anchor_atoms": []}
    _, index, step = max(candidates, key=lambda x: (x[0], x[1]))
    return {"matched": True, "index": index, "candidate": render_call(step["call"]), "kind": "tool", "anchor_atoms": step["before"]}
