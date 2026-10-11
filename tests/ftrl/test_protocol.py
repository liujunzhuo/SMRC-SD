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

"""Behavioral checks for FTRL observation grounding and matching boundaries."""

import pytest

from agent_system.environments.env_package.ftrl.core import acquired_atoms, answer_verify, information_atom, match_reference, parse_calls, score_trace
from agent_system.environments.env_package.ftrl.execute import execute


def test_tool_bundle_and_string_arguments():
    calls = parse_calls('<think>plan</think><tool_call>{"name":"f","arguments":"{\\"x\\":2}"}</tool_call><tool_call>{"name":"g","arguments":{}}</tool_call>')
    assert calls == [{"name": "f", "arguments": {"x": 2}}, {"name": "g", "arguments": {}}]
    assert parse_calls("<tool_call>bad</tool_call>") == []


def test_fresh_namespace_and_errors():
    calls = [{"name": "counter", "arguments": {}}] * 2
    assert execute({"counter": "count=0\ndef counter():\n global count\n count+=1\n return count"}, calls) == ["1", "1"]
    assert "No code found" in execute({}, calls[:1])[0]
    assert "an error occured" in execute({"counter": "def counter(: pass"}, calls[:1])[0]


def test_score_deduplicates_solved_subgoals_but_counts_repeated_calls():
    task = {"unsolved_set": {"f": ["Paris"], "g": ["France"]}}
    event = {"calls": [{"name": "f"}, {"name": "f"}, {"name": "g"}], "outputs": ["Paris", "Paris", "France"]}
    scores = score_trace(task, [event])
    assert scores["solve_precision"] == pytest.approx(2 / 3)
    assert scores["solve_rate"] == 1
    assert scores["solve_f1"] == pytest.approx(4 / 5)
    assert task["unsolved_set"]["f"] == ["Paris"]
    assert score_trace(task, [])["solve_f1"] == 0
    assert answer_verify("Paris, FRANCE", "Paris, France")


def test_information_is_query_bound_order_invariant_and_error_aware():
    a = {"name": "population", "arguments": {"city": "A"}}
    b = {"name": "population", "arguments": {"city": "B"}}
    assert information_atom("2", a) != information_atom("2", b)
    events = [{"calls": [a], "outputs": ["2"]}, {"calls": [b], "outputs": ["2"]}]
    assert acquired_atoms(events) == acquired_atoms(events[::-1])
    assert len(acquired_atoms(events + events)) == 2
    assert information_atom('{"error":"not found"}', a) is None
    assert information_atom('{"message":"You have 3 cats."}', a) is not None
    assert information_atom('{"status":"Mpox emergency ended."}', a) is not None
    assert information_atom('{"status":"ok"}', a) is None


def test_query_equivalence_is_field_limited_and_does_not_change_calls():
    import copy

    a = {"name": "lookup", "arguments": {"q": "Oxford", "id": "AbC"}}
    b = {"name": "lookup", "arguments": {"q": "oxford", "id": "AbC"}}
    original = copy.deepcopy(a)
    fields = {"lookup": ["q"]}
    assert information_atom("result", a) != information_atom("result", b)
    assert information_atom("result", a, None, fields) == information_atom("result", b, None, fields)
    b["arguments"]["id"] = "abc"
    assert information_atom("result", a, None, fields) != information_atom("result", b, None, fields)
    assert a == original


@pytest.mark.parametrize(
    "code,expected",
    [
        ("def f(q):\n return q.lower()", ["q"]),
        ("def f(q):\n return data[q]", []),
        ("def f(q):\n return q.lower(), q", []),
        ("def f(q):\n if not q or not isinstance(q,str): return None\n return q.lower()", ["q"]),
        ('def f(q):\n x=q or "default"\n return q.lower(), x', []),
        ("def f(q):\n return q.lower(), locals()", []),
    ],
)
def test_query_lowering_static_positive_and_negative_uses(code, expected):
    from agent_system.environments.env_package.ftrl.query_equivalence import fields_for_code

    assert fields_for_code(code, "f") == expected


def test_subset_uses_supported_anchor_but_root_is_not_wildcard():
    ref = {"steps": [{"before": [], "after": ["a"], "call": {"name": "first", "arguments": {}}}, {"before": ["a"], "after": ["a", "b"], "call": {"name": "second", "arguments": {}}}], "final_atoms": ["a", "b"], "final_response": "<answer>done</answer>"}
    assert match_reference(ref, [], "exact")["index"] == 0
    assert not match_reference(ref, ["wrong"], "subset")["matched"]
    assert not match_reference(ref, ["a", "extra"], "exact")["matched"]
    assert match_reference(ref, ["a", "extra"], "subset")["index"] == 1
    assert match_reference(ref, ["b", "a"], "exact")["kind"] == "completion"


def test_executor_preserves_upstream_exec_local_namespace():
    from agent_system.environments.env_package.ftrl.execute import execute

    # exec-local helpers are not visible in a tool function's globals upstream.
    code = 'import re\ndef lookup():\n    return re.sub("x", "y", "x")'
    out = execute({"lookup": code}, [{"name": "lookup", "arguments": {}}])[0]
    assert "name 're' is not defined" in out


def test_repeated_call_bundle_is_scored_instead_of_rejected(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from agent_system.environments.env_package.ftrl.service import Benchmark

    monkeypatch.setattr("agent_system.environments.env_package.ftrl.service.sys", SimpleNamespace(executable="/usr/bin/python3"))
    task = {"task_id": "train:x", "codes": {"lookup": "def lookup():\n    return 42"}, "unsolved_set": {"lookup": ["42"]}}
    (tmp_path / "train.jsonl").write_text(json.dumps(task) + "\n")
    (tmp_path / "test.jsonl").write_text("")
    server = Benchmark(tmp_path)
    calls = [{"name": "lookup", "arguments": {}}] * 65
    outputs = server.call({"operation": "execute", "task_id": "train:x", "calls": calls})
    assert outputs == ["42"] * 65
    scores = server.call({"operation": "score", "task_id": "train:x", "events": [{"calls": calls, "outputs": outputs}]})
    assert scores["calls"] == 65
    assert scores["solved"] == 1
    assert scores["solve_f1"] == 2 / 66


def test_literal_defaults_merge_omission_only_without_losing_query_identity():
    from copy import deepcopy

    from agent_system.environments.env_package.ftrl.state_defaults import extract_defaults

    defaults, skipped = extract_defaults({"lookup": 'def lookup(entity, date, unit="km", include_previous=False, filters=None, *, tags=[]): pass'})
    assert not skipped
    call = {"name": "lookup", "arguments": {"entity": "A", "date": "2024"}}
    explicit = {"name": "lookup", "arguments": {**call["arguments"], "unit": "km", "include_previous": False, "filters": None, "tags": []}}
    saved = deepcopy((defaults, call, explicit))
    assert information_atom("2", call, defaults) == information_atom("2", explicit, defaults)
    assert information_atom("2", call) != information_atom("2", explicit)
    for key, value in [("entity", "B"), ("date", "2025"), ("unit", "m"), ("include_previous", True), ("unexpected", 1)]:
        other = {"name": "lookup", "arguments": {**call["arguments"], key: value}}
        assert information_atom("2", call, defaults) != information_atom("2", other, defaults)
    assert information_atom("2", call, defaults) != information_atom("2", {**call, "name": "different"}, defaults)
    assert (defaults, call, explicit) == saved
    events = [{"calls": [call, explicit], "outputs": ["2", "2"]}]
    assert len(acquired_atoms(events, defaults)) == 1
    assert information_atom("Error: failed", call, defaults) is None


def test_dynamic_defaults_are_not_evaluated_and_missing_task_fails_closed(tmp_path):
    import json

    from agent_system.environments.env_package.ftrl.state_defaults import STATE_MODE, extract_defaults, load_state_defaults, task_defaults

    defaults, skipped = extract_defaults({"lookup": "def lookup(x=SIDE_EFFECT(), y=(1, 2), z=-2): pass"})
    assert defaults == {"lookup": {"z": -2}}
    assert len(skipped) == 2
    path = tmp_path / "defaults.json"
    path.write_text(json.dumps({"state_mode": STATE_MODE, "tasks": {"train:x": defaults}}))
    artifact = load_state_defaults(path)
    assert task_defaults(artifact, "train:x") == defaults
    with pytest.raises(KeyError):
        task_defaults(artifact, "missing")
    assert task_defaults(None, "anything") is None


def test_normalized_reference_and_student_states_share_exact_subset_boundaries():
    defaults = {"lookup": {"unit": "km"}}
    ref_call = {"name": "lookup", "arguments": {"entity": "A"}}
    student_call = {"name": "lookup", "arguments": {"entity": "A", "unit": "km"}}
    after = acquired_atoms([{"calls": [ref_call], "outputs": ["2"]}], defaults)
    current = acquired_atoms([{"calls": [student_call], "outputs": ["2"]}], defaults)
    ref = {"steps": [{"before": [], "after": after, "call": ref_call}], "final_atoms": after, "final_response": "<answer>done</answer>"}
    assert match_reference(ref, current, "exact")["kind"] == "completion"
    assert not match_reference(ref, current + ["extra"], "exact")["matched"]
    assert match_reference(ref, current + ["extra"], "subset")["kind"] == "completion"


def test_shared_student_instruction_preserves_question_and_rejects_unknown_prefix():
    from copy import deepcopy

    from agent_system.environments.env_package.ftrl.prompts import ANSWER_INSTRUCTION, TOOL_INSTRUCTION, student_messages

    question = 'What does "Question: X" mean?\nPreserve this line exactly.'
    train = [{"role": "user", "content": TOOL_INSTRUCTION + ANSWER_INSTRUCTION + "\nQuestion: " + question}]
    test = [{"role": "user", "content": TOOL_INSTRUCTION + "\nQuestion: " + question}]
    saved = deepcopy((train, test))
    a, b = student_messages(train, "concise_shared_v1"), student_messages(test, "concise_shared_v1")
    assert a == b
    assert a[0]["content"].partition("\nQuestion: ")[2] == question
    assert a[0]["content"].count("Only output the final answer") == 1
    assert (train, test) == saved
    assert student_messages(train) == train
    with pytest.raises(ValueError, match="Unexpected FTRL instruction"):
        student_messages([{"role": "user", "content": "different instruction\nQuestion: x"}], "concise_shared_v1")


def test_continuation_keeps_tool_outputs_and_mode_on_non_eos_response():
    from agent_system.environments.env_package.ftrl.prompts import tool_feedback

    outputs = ["full observation A", "full observation B"]
    for mode in (True, False):
        for ended in (True, False):
            feedback = tool_feedback(outputs, ended, mode)
            assert feedback.startswith("\n<|im_start|>user\n" if ended else "<|im_end|>\n<|im_start|>user\n")
            assert all(f"<tool_response>\n{x}\n</tool_response>" in feedback for x in outputs)
            assert feedback.endswith("<|im_start|>assistant\n" if mode else "<|im_start|>assistant\n<think>\n\n</think>\n\n")
