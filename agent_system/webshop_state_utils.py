# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

"""Pure WebShop oracle-path and progress-state helpers.

This module deliberately does not import Ray, Gym, or WebShop itself.  It is
shared by the environment worker, online SMRC-SD teacher, and offline probe so
their state semantics cannot silently drift apart.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def normalize_webshop_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def webshop_task_identity_payload(goal: dict[str, Any] | None) -> dict[str, Any]:
    """Return the canonical, worker-independent identity of a WebShop goal."""
    goal = dict(goal or {})
    return {
        "instruction_text": " ".join(str(goal.get("instruction_text") or "").split()),
        "target_asin": str(goal.get("asin") or "").strip().upper(),
        "target_options": {
            normalize_webshop_text(key): normalize_webshop_text(value)
            for key, value in sorted(dict(goal.get("goal_options") or {}).items())
        },
    }


def webshop_task_id(goal: dict[str, Any] | None, *, goal_index: int | None = None) -> str:
    """Build a stable semantic id, optionally disambiguated by canonical index."""
    payload = webshop_task_identity_payload(goal)
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    task_id = f"webshop:{digest}"
    if goal_index is not None:
        task_id = f"{task_id}:{int(goal_index):05d}"
    return task_id


def build_webshop_oracle_manifest_record(
    goal: dict[str, Any],
    *,
    goal_index: int,
    query_variant: str,
    expected_score: float | None = None,
) -> dict[str, Any]:
    """Create one auditable oracle record in the canonical goal order."""
    reference = build_webshop_oracle_reference(goal, query_variant=query_variant)
    identity = webshop_task_identity_payload(goal)
    return {
        "goal_index": int(goal_index),
        "webshop_task_id": webshop_task_id(goal, goal_index=goal_index),
        **identity,
        "query_variant": query_variant,
        "full_actions": list(reference.get("full_actions") or []),
        "reference_states": list(reference.get("reference_states") or []),
        "expected_score": expected_score,
    }


def load_webshop_jsonl_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load an oracle JSONL manifest and fail on duplicate indices or task ids."""
    records: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    seen_task_ids: set[str] = set()
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            goal_index = int(record["goal_index"])
            task_id = str(record["webshop_task_id"])
            if goal_index in seen_indices:
                raise ValueError(f"Duplicate WebShop goal_index={goal_index} in {path}:{line_number}")
            if task_id in seen_task_ids:
                raise ValueError(f"Duplicate WebShop task id={task_id} in {path}:{line_number}")
            seen_indices.add(goal_index)
            seen_task_ids.add(task_id)
            records.append(record)
    records.sort(key=lambda record: int(record["goal_index"]))
    if [int(record["goal_index"]) for record in records] != list(range(len(records))):
        raise ValueError(f"WebShop oracle manifest indices must be contiguous from zero: {path}")
    return records


def webshop_action_argument(action: str) -> str:
    match = re.match(r"^(?:search|click)\[(.*)\]$", str(action), re.IGNORECASE | re.DOTALL)
    return normalize_webshop_text(match.group(1)) if match else ""


def webshop_page_type(url: str) -> str:
    if "/search_results/" in str(url or ""):
        return "search_results"
    if "/item_page/" in str(url or ""):
        return "item_page"
    if "/item_sub_page/" in str(url or ""):
        return "item_sub_page"
    if "/done/" in str(url or ""):
        return "done"
    return "search_home"


def normalize_webshop_state(state: dict[str, Any] | None) -> dict[str, Any]:
    state = dict(state or {})
    keywords = state.get("keywords") or []
    if isinstance(keywords, str):
        keywords = keywords.split()
    return {
        "page_type": normalize_webshop_text(state.get("page_type")) or "unknown",
        "keywords": [normalize_webshop_text(token) for token in keywords if normalize_webshop_text(token)],
        "page": state.get("page"),
        "asin": normalize_webshop_text(state.get("asin")),
        "options": {
            normalize_webshop_text(key): normalize_webshop_text(value)
            for key, value in dict(state.get("options") or {}).items()
        },
    }


def capture_webshop_state(env: Any) -> dict[str, Any]:
    """Capture the simulator state for the currently rendered page."""
    session = env.server.user_sessions.get(env.session, {})
    return normalize_webshop_state(
        {
            "page_type": webshop_page_type(env.browser.current_url),
            "keywords": list(session.get("keywords") or []),
            "page": session.get("page"),
            "asin": str(session.get("asin") or ""),
            "options": dict(session.get("options") or {}),
        }
    )


def webshop_progress_state_key(state: dict[str, Any] | None) -> tuple[Any, ...]:
    """Probe-validated state key that ignores search wording and page number."""
    normalized = normalize_webshop_state(state)
    return (
        normalized["page_type"],
        normalized["asin"],
        tuple(sorted(normalized["options"].items())),
    )


def webshop_strict_state_key(state: dict[str, Any] | None) -> tuple[Any, ...]:
    normalized = normalize_webshop_state(state)
    return webshop_progress_state_key(normalized) + (tuple(normalized["keywords"]), normalized["page"])


def webshop_state_component_match(current: dict[str, Any] | None, reference: dict[str, Any] | None) -> dict[str, bool]:
    current_normalized = normalize_webshop_state(current)
    reference_normalized = normalize_webshop_state(reference)
    return {
        "page_type": current_normalized["page_type"] == reference_normalized["page_type"],
        "asin": current_normalized["asin"] == reference_normalized["asin"],
        "options": current_normalized["options"] == reference_normalized["options"],
    }


def webshop_action_is_admissible(action: str, available_actions: Any) -> bool:
    """Check a concrete WebShop action against raw or prompt action lists."""
    action_text = normalize_webshop_text(action)
    argument = webshop_action_argument(action)
    if isinstance(available_actions, dict):
        if action_text.startswith("search["):
            return bool(available_actions.get("has_search_bar")) and bool(argument)
        if action_text.startswith("click["):
            clickables = {normalize_webshop_text(value) for value in available_actions.get("clickables", [])}
            return bool(argument) and argument in clickables
        return False

    normalized_available = [normalize_webshop_text(value) for value in list(available_actions or [])]
    if action_text.startswith("search["):
        return bool(argument) and any(value == "search[<your query>]" or value == action_text for value in normalized_available)
    return action_text in normalized_available


def webshop_goal_progress_guidance(
    current_state: dict[str, Any] | None,
    *,
    target_asin: str,
    target_options: dict[str, Any] | None,
    available_actions: Any,
    reference_query: str = "",
) -> dict[str, Any]:
    """Build order-invariant, goal-relative guidance for one WebShop state.

    Canonical WebShop paths serialize option clicks in one arbitrary order, but
    the simulator accepts any order.  This matcher therefore treats selected
    target options as a set of satisfied goal constraints.  It exposes a
    concrete next action only when the compatible action is unique; otherwise
    it returns set-valued guidance and lets the teacher choose from the current
    admissible actions.
    """
    state = normalize_webshop_state(current_state)
    normalized_asin = normalize_webshop_text(target_asin)
    normalized_options = {
        normalize_webshop_text(key): normalize_webshop_text(value)
        for key, value in dict(target_options or {}).items()
    }
    selected_options = dict(state["options"])
    options_compatible = all(
        key in normalized_options and normalized_options[key] == value
        for key, value in selected_options.items()
    )
    base = {
        "action": "",
        "candidate_actions": [],
        "confidence": 0.0,
        "current_state": state,
        "guidance_kind": "none",
        "reference_query": str(reference_query or "").strip(),
        "remaining_options": {},
        "selected_options": selected_options,
        "state_match": False,
        "target_asin": str(target_asin or "").strip(),
        "target_options": normalized_options,
        "use_action": False,
        "state_component_match": {
            "page_type": state["page_type"] in {"search_home", "search_results", "item_page"},
            "asin": not state["asin"] or state["asin"] == normalized_asin,
            "options": options_compatible,
        },
    }
    if not normalized_asin:
        return {**base, "route": "webshop_goal_progress_missing_target"}

    page_type = state["page_type"]
    if page_type == "search_home":
        concrete_query = str(reference_query or normalized_asin).strip()
        if not concrete_query or not webshop_action_is_admissible(f"search[{concrete_query}]", available_actions):
            return {**base, "route": "webshop_goal_progress_no_search_action"}
        return {
            **base,
            "route": "webshop_goal_progress_search_compatible",
            "confidence": 1.0,
            "guidance_kind": "search_query_set",
            "state_match": True,
        }

    if page_type == "search_results":
        action = f"click[{target_asin.strip()}]"
        if not webshop_action_is_admissible(action, available_actions):
            # Leave unmatched: the student can page/re-search from the observation.
            return {**base, "route": "webshop_goal_progress_target_not_visible"}
        return {
            **base,
            "action": action,
            "candidate_actions": [action],
            "confidence": 1.0,
            "guidance_kind": "unique_action",
            "route": "webshop_goal_progress_target_visible",
            "state_match": True,
            "use_action": True,
        }

    # Wrong-ASIN / incompatible option pages stay unmatched (MatchOnly abstains).
    if page_type != "item_page" or state["asin"] != normalized_asin or not options_compatible:
        return {**base, "route": "webshop_goal_progress_incompatible"}

    remaining_options = {
        key: value for key, value in normalized_options.items() if selected_options.get(key) != value
    }
    option_actions = list(
        dict.fromkeys(f"click[{value}]" for _, value in sorted(remaining_options.items()))
    )
    if option_actions and not all(
        webshop_action_is_admissible(action, available_actions) for action in option_actions
    ):
        return {
            **base,
            "remaining_options": remaining_options,
            "route": "webshop_goal_progress_required_option_not_admissible",
        }
    if len(remaining_options) > 1:
        return {
            **base,
            "candidate_actions": option_actions,
            "confidence": 1.0,
            "guidance_kind": "option_set",
            "remaining_options": remaining_options,
            "route": "webshop_goal_progress_options_compatible",
            "state_match": True,
        }
    if len(remaining_options) == 1:
        return {
            **base,
            "action": option_actions[0],
            "candidate_actions": option_actions,
            "confidence": 1.0,
            "guidance_kind": "unique_action",
            "remaining_options": remaining_options,
            "route": "webshop_goal_progress_one_option_remaining",
            "state_match": True,
            "use_action": True,
        }

    buy_action = "click[buy now]"
    if not webshop_action_is_admissible(buy_action, available_actions):
        return {
            **base,
            "remaining_options": {},
            "route": "webshop_goal_progress_buy_not_admissible",
        }
    return {
        **base,
        "action": buy_action,
        "candidate_actions": [buy_action],
        "confidence": 1.0,
        "guidance_kind": "unique_action",
        "remaining_options": {},
        "route": "webshop_goal_progress_ready_to_buy",
        "state_match": True,
        "use_action": True,
    }


def _oracle_query(goal: dict[str, Any], variant: str) -> str:
    full_name = " ".join(str(goal.get("name") or "").replace("[", " ").replace("]", " ").split())
    if variant == "full_name":
        return full_name
    if variant != "short_name":
        raise ValueError(f"Unsupported WebShop oracle query variant: {variant}")
    short_name = " ".join(full_name.split(",", 1)[0].split())
    if normalize_webshop_text(short_name) == normalize_webshop_text(full_name):
        words = full_name.split()
        short_name = " ".join(words[: max(4, min(10, len(words) // 2))])
    return short_name or full_name


def build_webshop_oracle_reference(goal: dict[str, Any], *, query_variant: str = "short_name") -> dict[str, Any]:
    """Construct WebShop's hidden-goal successful path and pre-action states."""
    goal = dict(goal or {})
    asin = str(goal.get("asin") or "").strip()
    query = _oracle_query(goal, query_variant)
    goal_options = [(str(key), str(value)) for key, value in dict(goal.get("goal_options") or {}).items()]
    if not asin or not query:
        return {"full_actions": [], "reference_states": [], "query_variant": query_variant}

    actions = [
        f"search[{query}]",
        f"click[{asin}]",
        *(f"click[{value}]" for _, value in goal_options),
        "click[buy now]",
    ]
    state = normalize_webshop_state({"page_type": "search_home"})
    reference_states: list[dict[str, Any]] = []
    for action_index, _action in enumerate(actions):
        reference_states.append(
            {
                **state,
                "keywords": list(state["keywords"]),
                "options": dict(state["options"]),
            }
        )
        if action_index == 0:
            state = normalize_webshop_state(
                {
                    "page_type": "search_results",
                    "keywords": query.split(),
                    "page": 1,
                }
            )
        elif action_index == 1:
            state = normalize_webshop_state(
                {
                    **state,
                    "page_type": "item_page",
                    "asin": asin,
                    "options": {},
                }
            )
        elif action_index < len(actions) - 1:
            option_key, option_value = goal_options[action_index - 2]
            state = normalize_webshop_state(
                {
                    **state,
                    "page_type": "item_page",
                    "asin": asin,
                    "options": {**state["options"], option_key: option_value},
                }
            )

    return {
        "full_actions": actions,
        "reference_states": reference_states,
        "query_variant": query_variant,
        "target_asin": asin,
        "target_options": {key: value for key, value in goal_options},
    }
