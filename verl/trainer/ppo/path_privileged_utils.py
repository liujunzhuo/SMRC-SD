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

"""Path-privileged teacher context construction for agent OPD."""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from agent_system.webshop_state_utils import (
    normalize_webshop_state,
    webshop_action_argument,
    webshop_action_is_admissible,
    webshop_goal_progress_guidance,
    webshop_progress_state_key,
    webshop_state_component_match,
)
from verl import DataProto
from verl.trainer.ppo.rlsd_utils import SkillProvider
from verl.utils.model import compute_position_id_with_mask


def _as_py(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list, tuple)):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _as_list(value: Any) -> List[Any]:
    value = _as_py(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def normalize_actions(actions: Any) -> List[str]:
    return [str(action).strip() for action in _as_list(actions) if str(action).strip()]


def normalize_action_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def normalize_observation_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def is_truthy(value: Any, default: bool = True) -> bool:
    value = _as_py(value)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "valid"}:
            return True
        if text in {"0", "false", "no", "n", "invalid", ""}:
            return False
    return bool(value)


def is_clean_executed_action(action: Any) -> bool:
    """Return whether an executed action is suitable for state/history inference.

    ALFWorld dumps may contain parser fragments such as ``"foo\\n\\naction: bar"``
    or markup tags when the model failed to emit a clean ``<action>`` block.
    Such strings should not be used as environment state evidence.
    """
    text = str(_as_py(action) or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if "\n" in text or "\r" in text or "<" in text or ">" in text or "```" in text:
        return False
    if "action:" in lowered or "assistant" in lowered or "user" in lowered or "person" in lowered:
        return False
    if len(text) > 120:
        return False
    return True


def compact_action_text(action: Any, max_chars: int = 80) -> str:
    text = re.sub(r"\s+", " ", str(_as_py(action) or "").strip())
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def extract_admissible_actions(prompt_text: str) -> List[str]:
    """Extract the current admissible action list from a decoded prompt.

    Two serialization formats must be supported:

    - **ALFWorld** uses a numpy-style repr: single-quoted actions separated by
      spaces (no commas), optionally several per line, wrapped in ``[`` ... ``]``
      with the first action sharing the preamble line and the last followed by
      ``].``. ALFWorld actions never contain apostrophes.
    - **WebShop** uses a Python list repr: one comma-separated single-quoted
      action per line. Its values may embed apostrophes (for example
      ``20''x20''``) and square brackets (for example ``click[...]``).

    A single greedy regex breaks ALFWorld (it glues the space-separated actions
    on one line together) and a single per-quote regex breaks WebShop (it splits
    values containing apostrophes). We therefore isolate the bracketed list body
    and dispatch on whether the list is comma-separated (WebShop) or
    space-separated (ALFWorld).
    """
    if not prompt_text:
        return []
    lower = prompt_text.lower()
    marker = "your admissible actions"
    start = lower.rfind(marker)
    if start < 0:
        return []
    segment = prompt_text[start:]
    end_marker = "\n\nNow it's your turn"
    end = segment.find(end_marker)
    if end >= 0:
        segment = segment[:end]
    # Restrict to the bracketed list body so the preamble ("... are:") and the
    # trailing "]." punctuation cannot leak into or truncate parsing.
    open_idx = segment.find("[")
    close_idx = segment.rfind("]")
    body = segment[open_idx + 1 : close_idx] if 0 <= open_idx < close_idx else segment
    # A closing quote immediately followed by a comma only appears in the
    # comma-separated WebShop repr; ALFWorld's numpy repr is space-separated.
    if re.search(r"'\s*,", body):
        actions: List[str] = []
        for line in body.splitlines():
            item = line.strip().rstrip(",").strip()
            if len(item) >= 2 and item[0] == "'" and item[-1] == "'":
                actions.append(item[1:-1])
        return [action.strip() for action in actions if action.strip()]
    return [action.strip() for action in re.findall(r"'([^']*)'", body) if action.strip()]


def extract_current_observation(prompt_text: str) -> str:
    if not prompt_text:
        return ""
    patterns = [
        r"You are now at step\s+\d+\s+and your current observation is:\s*(.*?)(?:\nYour admissible actions|\n\nYour admissible actions)",
        r"Your current observation is:\s*(.*?)(?:\n\nYour task|\nYour task|\nYour admissible actions|\n\nYour admissible actions)",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt_text, re.DOTALL)
        if match:
            return str(match.group(1) or "").strip()
    return ""


def extract_task_text(prompt_text: str) -> str:
    if not prompt_text:
        return ""
    patterns = [
        r"Your task is to:\s*(.*?)(?:\.?\n|$)",
        r"task is to:\s*(.*?)(?:\.?\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt_text, re.IGNORECASE | re.DOTALL)
        if match:
            return re.sub(r"\s+", " ", str(match.group(1) or "").strip().rstrip("."))
    return ""


def infer_location_from_observation(observation: str) -> str:
    text = str(observation or "").strip()
    patterns = [
        r"You arrive at ([^.]+)\.",
        r"You are at ([^.]+)\.",
        r"You are in ([^.]+)\.",
        r"On the ([^,\.]+)",
        r"The ([^,\.]+) is (?:open|closed)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", str(match.group(1) or "").strip())
    return "unknown"


def parse_walkthrough_preview(metadata: Dict[str, Any]) -> List[str]:
    preview = metadata.get("privileged_context_preview") or metadata.get("walkthrough_preview")
    lines = normalize_actions(preview)
    if not lines:
        return []
    candidate = lines[-1]
    if "->" not in candidate:
        return []
    return normalize_actions([part.strip() for part in candidate.split("->")])


def extract_full_actions(record: Dict[str, Any]) -> List[str]:
    metadata = record.get("metadata") or {}
    for key in ("full_actions", "trajectory_actions", "actions", "oracle_actions", "walkthrough_actions"):
        actions = normalize_actions(record.get(key) or metadata.get(key))
        if actions:
            return actions
    return parse_walkthrough_preview(metadata)


def _metadata_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    actions = extract_full_actions(record)
    if actions and "full_actions" not in metadata:
        metadata["full_actions"] = actions
    return metadata


def relocate_alfworld_path(path: Optional[str]) -> Optional[str]:
    if not path or not isinstance(path, str):
        return path
    expanded = os.path.expanduser(path)
    if os.path.exists(expanded):
        return expanded
    marker = "json_2.1.1"
    if marker not in expanded:
        return expanded
    alfworld_data = os.environ.get("ALFWORLD_DATA")
    if not alfworld_data:
        return expanded
    suffix = expanded.split(marker, 1)[1].lstrip(os.sep)
    relocated = os.path.join(os.path.abspath(os.path.expanduser(alfworld_data)), marker, suffix)
    return relocated if os.path.exists(relocated) else expanded


def alfworld_path_lookup_keys(path: Optional[str]) -> List[str]:
    """Stable ALFWorld lookup keys that survive absolute/relative path rewriting."""
    if not path or not isinstance(path, str):
        return []
    keys = [str(path)]
    marker = "json_2.1.1"
    if marker in path:
        suffix = path.split(marker, 1)[1].lstrip(os.sep)
        keys.append(f"{marker}/{suffix}")
    relocated = relocate_alfworld_path(path)
    if relocated:
        keys.append(str(relocated))
        if marker in str(relocated):
            suffix = str(relocated).split(marker, 1)[1].lstrip(os.sep)
            keys.append(f"{marker}/{suffix}")
    # Preserve order while dropping duplicates.
    ordered: List[str] = []
    seen = set()
    for key in keys:
        if key and key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


@dataclass
class WalkthroughState:
    step_index: int
    observation: str
    next_observation: str
    admissible_actions: List[str]
    next_action: str
    action_history: List[str]
    observation_norm: str = ""
    next_action_norm: str = ""
    action_history_norm: List[str] = field(default_factory=list)


class WalkthroughTraceCache:
    """CPU TextWorld walkthrough replay cache for ALFWorld state-aligned recovery."""

    def __init__(self, enabled: bool = True, retain_diagnostics: bool = True):
        self.enabled = enabled
        self.retain_diagnostics = retain_diagnostics
        self._cache: Dict[Tuple[str, Tuple[str, ...]], List[WalkthroughState]] = {}
        self._errors: Counter = Counter()
        self._build_count = 0

    @property
    def error_count(self) -> int:
        return int(sum(self._errors.values()))

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    @property
    def build_count(self) -> int:
        return self._build_count

    def get(self, gamefile: Optional[str], walkthrough_actions: List[str]) -> List[WalkthroughState]:
        if not self.enabled or not gamefile or not walkthrough_actions:
            return []
        relocated = relocate_alfworld_path(str(gamefile))
        if not relocated:
            return []
        key = (str(relocated), tuple(str(action) for action in walkthrough_actions))
        if key in self._cache:
            return self._cache[key]
        try:
            trace = self._build_trace(str(relocated), walkthrough_actions)
        except Exception as exc:
            self._errors[type(exc).__name__] += 1
            self._cache[key] = []
            return []
        self._build_count += 1
        self._cache[key] = trace
        return trace

    def _build_trace(self, gamefile: str, walkthrough_actions: List[str]) -> List[WalkthroughState]:
        import textworld
        import textworld.gym
        from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env_id = textworld.gym.register_game(gamefile, request_infos, wrappers=[AlfredDemangler(), AlfredInfos])
        env = textworld.gym.make(env_id)
        trace: List[WalkthroughState] = []
        try:
            obs, info = env.reset()
            history: List[str] = []
            for idx, action in enumerate(walkthrough_actions):
                admissible = normalize_actions((info or {}).get("admissible_commands")) if self.retain_diagnostics else []
                obs_before = str(obs or "")
                obs, _score, done, info = env.step(action)
                trace.append(
                    WalkthroughState(
                        step_index=idx,
                        observation=obs_before,
                        next_observation=str(obs or "") if self.retain_diagnostics else "",
                        admissible_actions=admissible if self.retain_diagnostics else [],
                        next_action=str(action or "").strip(),
                        action_history=list(history),
                        observation_norm=normalize_observation_text(obs_before),
                        next_action_norm=normalize_action_text(action),
                        action_history_norm=[normalize_action_text(item) for item in history],
                    )
                )
                history.append(str(action or "").strip())
                if done:
                    break
        finally:
            try:
                env.close()
            except Exception:
                pass
        return trace


@dataclass
class PathTrajectoryIndex:
    """Lookup table from ALFWorld gamefile/task key to successful action paths."""

    by_key: Dict[str, List[str]] = field(default_factory=dict)
    source_counts: Counter = field(default_factory=Counter)

    @classmethod
    def from_jsonl(cls, path: Optional[str]) -> "PathTrajectoryIndex":
        index = cls()
        if not path:
            return index
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path-OPD trajectory_index_path does not exist: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                actions = extract_full_actions(record)
                if not actions:
                    continue
                keys = []
                task_key = record.get("task_key")
                if task_key:
                    keys.append(str(task_key))
                webshop_task_id = record.get("webshop_task_id")
                if webshop_task_id:
                    keys.append(str(webshop_task_id))
                reset_kwargs = record.get("reset_kwargs") or {}
                gamefile = record.get("gamefile") or reset_kwargs.get("gamefile")
                if gamefile:
                    keys.append(str(gamefile))
                trajectory_id = record.get("trajectory_id") or record.get("prefix_id")
                if trajectory_id:
                    keys.append(str(trajectory_id))
                # WebShop oracle manifests also expose the canonical goal index as a
                # stable lookup key used by some reset_kwargs/session paths.
                if record.get("goal_index") is not None:
                    keys.append(str(int(record["goal_index"])))
                for key in keys:
                    for lookup_key in alfworld_path_lookup_keys(str(key)):
                        index.by_key.setdefault(lookup_key, actions)
                index.source_counts[str(record.get("source", "unknown"))] += 1
        return index

    def get(self, *keys: Optional[str]) -> List[str]:
        for key in keys:
            if not key:
                continue
            for lookup_key in alfworld_path_lookup_keys(str(key)):
                if lookup_key in self.by_key:
                    return list(self.by_key[lookup_key])
        return []


@dataclass
class PathCandidate:
    actions: List[str]
    source: str
    insert_step: int = 0
    prefix_id: str = ""
    source_traj_uid: str = ""


@dataclass
class PathTrajectoryBufferIndex:
    """Reloadable lookup table for student success paths saved in VMPR buffer."""

    path: Optional[str] = None
    source_filter: str = "student_autonomous"
    by_key: Dict[str, List[PathCandidate]] = field(default_factory=dict)
    source_counts: Counter = field(default_factory=Counter)
    _mtime: Optional[float] = None
    _reload_count: int = 0
    _last_stat_time: float = 0.0
    _stat_ttl_seconds: float = 0.25

    def maybe_reload(self) -> None:
        if not self.path:
            return
        path = os.path.expanduser(self.path)
        if not os.path.exists(path):
            self.by_key = {}
            self.source_counts = Counter()
            self._mtime = None
            return
        now = time.monotonic()
        if self._mtime is not None and now - self._last_stat_time < self._stat_ttl_seconds:
            return
        self._last_stat_time = now
        mtime = os.path.getmtime(path)
        if self._mtime is not None and mtime == self._mtime:
            return
        self._mtime = mtime
        self._reload_count += 1
        self.by_key = {}
        self.source_counts = Counter()
        seen: set[tuple[str, tuple[str, ...]]] = set()
        allowed_sources = {item.strip() for item in str(self.source_filter or "").split(",") if item.strip()}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                source = str(record.get("source") or "")
                if allowed_sources and source not in allowed_sources:
                    continue
                actions = extract_full_actions(record)
                if not actions:
                    continue
                keys: List[str] = []
                task_key = record.get("task_key")
                if task_key:
                    keys.append(str(task_key))
                reset_kwargs = record.get("reset_kwargs") or {}
                gamefile = record.get("gamefile") or reset_kwargs.get("gamefile")
                if gamefile:
                    keys.append(str(gamefile))
                prefix_id = str(record.get("prefix_id") or "")
                if prefix_id:
                    keys.append(prefix_id)
                metadata = record.get("metadata") or {}
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                metadata = metadata if isinstance(metadata, dict) else {}
                candidate = PathCandidate(
                    actions=list(actions),
                    source=source or "buffer",
                    insert_step=int(record.get("insert_step") or 0),
                    prefix_id=prefix_id,
                    source_traj_uid=str(metadata.get("source_traj_uid") or metadata.get("traj_uid") or ""),
                )
                for key in keys:
                    dedup_key = (str(key), tuple(candidate.actions))
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    self.by_key.setdefault(str(key), []).append(candidate)
                    relocated_key = relocate_alfworld_path(str(key))
                    if relocated_key and relocated_key != key:
                        relocated_dedup_key = (str(relocated_key), tuple(candidate.actions))
                        if relocated_dedup_key not in seen:
                            seen.add(relocated_dedup_key)
                            self.by_key.setdefault(str(relocated_key), []).append(candidate)
                self.source_counts[source or "buffer"] += 1

    def get(self, *keys: Optional[str], max_candidates: int = 16) -> List[PathCandidate]:
        self.maybe_reload()
        candidates: List[PathCandidate] = []
        seen: set[tuple[str, ...]] = set()
        for key in keys:
            if not key:
                continue
            lookup_keys = [str(key)]
            relocated_key = relocate_alfworld_path(str(key))
            if relocated_key and str(relocated_key) not in lookup_keys:
                lookup_keys.append(str(relocated_key))
            for lookup_key in lookup_keys:
                for candidate in self.by_key.get(lookup_key, []):
                    action_key = tuple(candidate.actions)
                    if action_key in seen:
                        continue
                    seen.add(action_key)
                    candidates.append(candidate)
                    if len(candidates) >= max_candidates:
                        return candidates
        return candidates

    @property
    def candidate_count(self) -> int:
        seen: set[tuple[str, ...]] = set()
        for candidates in self.by_key.values():
            for candidate in candidates:
                seen.add(tuple(candidate.actions))
        return len(seen)

    @property
    def reload_count(self) -> int:
        return self._reload_count


class PathPrivilegedContextProvider:
    """Builds privileged context blocks for same-model OPD teachers."""

    PATH_CONTEXTS = {
        "gt_path",
        "gt_recovery_aligned",
        "skill_gt_path",
        "skill_gt_recovery_aligned",
        "state_gt_path",
        "state_gt_recovery_aligned",
    }
    RECOVERY_CONTEXTS = {"gt_recovery_aligned", "skill_gt_recovery_aligned", "state_gt_recovery_aligned"}
    STATE_CONTEXTS = {"state_history", "state_gt_path", "state_gt_recovery_aligned"}
    SKILL_CONTEXTS = {"skill", "skill_gt_path", "skill_gt_recovery_aligned"}

    def __init__(self, *, config, skill_provider: Optional[SkillProvider] = None):
        path_cfg = config.algorithm.get("path_opd", {}) if hasattr(config, "algorithm") else {}
        self.teacher_context = str(path_cfg.get("teacher_context", "gt_path"))
        self.strict_gt_path = bool(path_cfg.get("strict_gt_path", True))
        self.max_remaining_actions = int(path_cfg.get("max_remaining_actions", 20))
        self.include_executed_prefix = bool(path_cfg.get("include_executed_prefix", True))
        self.include_remaining_path = bool(path_cfg.get("include_remaining_path", True))
        self.include_full_path = bool(path_cfg.get("include_full_path", True))
        self.path_source_policy = str(path_cfg.get("path_source_policy", "canonical") or "canonical").lower()
        if self.path_source_policy not in {
            "canonical",
            "prefer_in_batch_success",
            "in_batch_success_only",
            "prefer_buffer_success",
            "buffer_success_only",
            "prefer_student_success",
            "best_aligned_success",
            "best_state_aligned_success",
        }:
            raise ValueError(
                f"Unsupported algorithm.path_opd.path_source_policy={self.path_source_policy}. "
                "Expected canonical, prefer_in_batch_success, in_batch_success_only, prefer_buffer_success, "
                "buffer_success_only, prefer_student_success, best_aligned_success, or best_state_aligned_success."
            )
        self.path_buffer_index = PathTrajectoryBufferIndex(
            path=path_cfg.get("path_buffer_path"),
            source_filter=str(path_cfg.get("path_buffer_source_filter", "student_autonomous") or "student_autonomous"),
        )
        self.path_buffer_max_candidates = int(path_cfg.get("path_buffer_max_candidates", 16) or 16)
        self.path_buffer_max_action_len = int(path_cfg.get("path_buffer_max_action_len", 0) or 0)
        self.path_buffer_max_length_ratio_to_canonical = float(path_cfg.get("path_buffer_max_length_ratio_to_canonical", 0.0) or 0.0)
        self.dont_reprompt_on_self_success = bool(path_cfg.get("dont_reprompt_on_self_success", False))
        self.filter_no_progress_success_actions = bool(path_cfg.get("filter_no_progress_success_actions", True))
        self.enable_history_suffix_alignment = bool(path_cfg.get("enable_history_suffix_alignment", False))
        self.alignment_mode = str(path_cfg.get("alignment_mode", "full_start")).lower()
        if self.alignment_mode not in {"full_start", "dynamic", "legacy"}:
            raise ValueError(f"Unsupported algorithm.path_opd.alignment_mode={self.alignment_mode}. Expected full_start, dynamic, or legacy.")
        self.enable_walkthrough_trace_recovery = bool(path_cfg.get("enable_walkthrough_trace_recovery", True))
        self.recovery_match_mode = str(path_cfg.get("recovery_match_mode", "history") or "history").lower()
        if self.recovery_match_mode not in {"history", "state"}:
            raise ValueError(
                f"Unsupported algorithm.path_opd.recovery_match_mode={self.recovery_match_mode!r}. "
                "Expected history or state."
            )
        self.recovery_state_domain = str(path_cfg.get("recovery_state_domain", "alfworld") or "alfworld").lower()
        if self.recovery_state_domain not in {"alfworld", "webshop"}:
            raise ValueError(
                f"Unsupported algorithm.path_opd.recovery_state_domain={self.recovery_state_domain!r}. "
                "Expected alfworld or webshop."
            )
        self.recovery_alfworld_object_match_mode = str(
            path_cfg.get("recovery_alfworld_object_match_mode", "exact") or "exact"
        ).lower()
        if self.recovery_alfworld_object_match_mode not in {"exact", "semantic_instance"}:
            raise ValueError(
                "Unsupported algorithm.path_opd.recovery_alfworld_object_match_mode="
                f"{self.recovery_alfworld_object_match_mode!r}. Expected exact or semantic_instance."
            )
        self.recovery_webshop_match_mode = str(
            path_cfg.get("recovery_webshop_match_mode", "canonical_exact") or "canonical_exact"
        ).lower()
        if self.recovery_webshop_match_mode not in {"canonical_exact", "goal_progress"}:
            raise ValueError(
                "Unsupported algorithm.path_opd.recovery_webshop_match_mode="
                f"{self.recovery_webshop_match_mode!r}. Expected canonical_exact or goal_progress."
            )
        self.state_sdl_selection = str(path_cfg.get("state_sdl_selection", "all") or "all").lower()
        if self.state_sdl_selection not in {
            "all",
            "match_only",
            "random_equal",
            "mismatch_only",
            "unsupported_equal",
        }:
            raise ValueError(
                f"Unsupported algorithm.path_opd.state_sdl_selection={self.state_sdl_selection!r}. "
                "Expected all, match_only, random_equal, mismatch_only, or unsupported_equal."
            )
        if self.state_sdl_selection != "all" and (
            self.recovery_match_mode not in {"history", "state"}
            or self.teacher_context not in self.RECOVERY_CONTEXTS
        ):
            raise ValueError(
                "Non-all algorithm.path_opd.state_sdl_selection requires a recovery teacher "
                "with algorithm.path_opd.recovery_match_mode in {history, state}."
            )
        self.state_sdl_random_seed = int(path_cfg.get("state_sdl_random_seed", 0) or 0)
        # state_match_context decouples "run the state matcher for row selection"
        # from "build a localized teacher prompt". When set to full_path, the
        # matcher still records recovery_state_match_ratio (so state_sdl_selection
        # row budgets remain identical) but the teacher receives the unconditional
        # full-path (D0) context. Combined with state_sdl_selection=match_only this
        # gives the D0@Matched control for the MatchOnly ablation.
        self.state_match_context = str(path_cfg.get("state_match_context", "recovery") or "recovery").lower()
        if self.state_match_context not in {"recovery", "full_path"}:
            raise ValueError(
                f"Unsupported algorithm.path_opd.state_match_context={self.state_match_context!r}. "
                "Expected recovery or full_path."
            )
        if self.state_match_context == "full_path" and (
            self.recovery_match_mode != "state" or self.teacher_context not in self.RECOVERY_CONTEXTS
        ):
            raise ValueError(
                "algorithm.path_opd.state_match_context=full_path requires a recovery teacher "
                "with algorithm.path_opd.recovery_match_mode=state."
            )
        self.recovery_prompt_style = str(path_cfg.get("recovery_prompt_style", "current") or "current").lower()
        if self.recovery_prompt_style not in {
            "current",
            "candidate_only",
            "candidate_prefix",
            "state_summary_candidate",
            "state_summary_only",
        }:
            raise ValueError(
                f"Unsupported algorithm.path_opd.recovery_prompt_style={self.recovery_prompt_style!r}. "
                "Expected current, candidate_only, candidate_prefix, state_summary_candidate, or state_summary_only."
            )
        self.recovery_guidance_style = str(
            path_cfg.get("recovery_guidance_style", "style_default") or "style_default"
        ).lower()
        if self.recovery_guidance_style not in {"style_default", "path_on_or_off"}:
            raise ValueError(
                f"Unsupported algorithm.path_opd.recovery_guidance_style={self.recovery_guidance_style!r}. "
                "Expected style_default or path_on_or_off."
            )
        if self.recovery_prompt_style in {
            "candidate_only",
            "candidate_prefix",
            "state_summary_candidate",
            "state_summary_only",
        } and (
            self.recovery_match_mode not in {"history", "state"}
            or self.teacher_context not in self.RECOVERY_CONTEXTS
            or self.state_match_context != "recovery"
            or self.recovery_state_domain not in {"alfworld", "webshop"}
        ):
            raise ValueError(
                f"algorithm.path_opd.recovery_prompt_style={self.recovery_prompt_style} requires a recovery teacher "
                "with recovery_state_domain in {{alfworld, webshop}}, recovery_match_mode in {{history, state}}, "
                "and state_match_context=recovery."
            )
        if self.path_source_policy == "best_state_aligned_success" and self.recovery_match_mode != "state":
            raise ValueError(
                "algorithm.path_opd.path_source_policy=best_state_aligned_success requires "
                "algorithm.path_opd.recovery_match_mode=state"
            )
        self.recovery_use_last_progress_observation_on_no_progress = bool(path_cfg.get("recovery_use_last_progress_observation_on_no_progress", False))
        # Correction-only is deliberately independent of the recovery matcher.
        # It is applied after selecting the best state/history-compatible
        # candidate, so it cannot expose a worse alternative merely because the
        # best candidate already agrees with the student. The older skip switch
        # remains a pre-selection filter for exact experiment reproducibility.
        self.recovery_correction_only = bool(path_cfg.get("recovery_correction_only", False))
        self.recovery_success_confirmation_sdl_weight = max(0.0, float(path_cfg.get("recovery_success_confirmation_sdl_weight", 1.0)))
        self.recovery_skip_if_action_matches_student = bool(path_cfg.get("recovery_skip_if_action_matches_student", False))
        self.include_invalid_action_history_in_state = bool(path_cfg.get("include_invalid_action_history_in_state", False))
        self.sdl_confidence_weighting = bool(path_cfg.get("sdl_confidence_weighting", False))
        default_weights = {
            "strict_start_next_oracle": 1.0,
            "strict_aligned_next_oracle": 1.0,
            "walkthrough_trace_obs_match_history_suffix": 1.0,
            "walkthrough_trace_obs_match_history_subsequence": 0.5,
            "walkthrough_trace_obs_match_history_mismatch": 0.0,
            "walkthrough_trace_obs_match_initial": 0.0,
            "walkthrough_state_compatible": 1.0,
            "walkthrough_state_candidate_matches_student": 0.0,
            "walkthrough_state_mismatch": 0.0,
            "walkthrough_trace_candidate_matches_student": 0.0,
            "strict_next_action_non_admissible": 0.0,
            "walkthrough_action_currently_admissible": 0.0,
            "walkthrough_seen_action_currently_admissible": 0.0,
            "other_full_gt_path": 0.0,
            "missing_gt_path": 0.0,
            "missing_admissible_actions": 0.0,
            "no_walkthrough_action_currently_admissible": 0.0,
        }
        configured_weights = path_cfg.get("sdl_confidence_weights", {}) or {}
        self.sdl_confidence_weights = {**default_weights, **{str(key): float(value) for key, value in dict(configured_weights).items()}}
        self.sdl_drop_repeated_action_after_turn = bool(path_cfg.get("sdl_drop_repeated_action_after_turn", False))
        self.sdl_repeat_drop_min_turn = int(path_cfg.get("sdl_repeat_drop_min_turn", 15))
        self.sdl_repeat_drop_weight = float(path_cfg.get("sdl_repeat_drop_weight", 0.0))
        self.policy_loss_filter = str(path_cfg.get("policy_loss_filter", "none") or "none").lower()
        if self.policy_loss_filter not in {"none", "repeat_drop", "repeat_after_turn"}:
            raise ValueError(
                f"Unsupported algorithm.path_opd.policy_loss_filter={self.policy_loss_filter!r}. "
                "Expected none or repeat_drop."
            )
        self.policy_loss_repeat_drop_weight = float(path_cfg.get("policy_loss_repeat_drop_weight", 0.0))
        self.skill_provider = skill_provider
        self.trajectory_index = PathTrajectoryIndex.from_jsonl(path_cfg.get("trajectory_index_path"))
        # Runtime recovery only needs the current reference observation and
        # action history.  Probe tooling constructs the cache with its default
        # diagnostic payload when next observations/admissible lists are needed.
        self.trace_cache = WalkthroughTraceCache(
            enabled=self.enable_walkthrough_trace_recovery,
            retain_diagnostics=False,
        )
        self.last_metrics: Dict[str, float] = {}
        self._candidate_filter_metrics: Dict[int, Dict[str, float]] = {}

    def _append_full_path(self, lines: List[str], full_path_text: str) -> None:
        if self.include_full_path:
            lines.append(self._full_path_heading())
            lines.append(full_path_text)

    def _full_path_heading(self) -> str:
        return "Complete successful path for this task:"

    @staticmethod
    def _append_path_guidance(lines: List[str]) -> None:
        lines.append("The current state may be on or off this path.")
        lines.append("Use the path as privileged guidance, but reason from the current observation and admissible actions.")

    @staticmethod
    def _append_aligned_guidance(lines: List[str]) -> None:
        lines.append("Use the matched path information as privileged guidance, but reason from the current observation and admissible actions.")

    def _append_matched_route_guidance(self, lines: List[str]) -> None:
        """Closing guidance for matched/localized routes.

        ``path_on_or_off`` freezes the payload blocks of the selected prompt style
        but swaps the assertive matched-path closing line for the D0 uncertain-path
        wording used by full-path fallback / D0@Matched.
        """
        if self.recovery_guidance_style == "path_on_or_off":
            self._append_path_guidance(lines)
        else:
            self._append_aligned_guidance(lines)

    @staticmethod
    def _webshop_options_text(options: Dict[str, Any]) -> str:
        if not options:
            return "(none)"
        return ", ".join(f"{key}={value}" for key, value in sorted(options.items()))

    def _append_webshop_goal_progress_guidance(self, lines: List[str], recovery: Dict[str, Any]) -> None:
        """Render simplified goal-progress blocks after the full successful path.

        Keeps a short state/progress summary plus unique or set-valued candidates.
        Avoids long procedural instructions that dilute chosen-token SDL.
        """
        state_signature = dict(recovery.get("state_signature") or {})
        selected_options = dict(recovery.get("selected_options") or {})
        remaining_options = dict(recovery.get("remaining_options") or {})
        reference_query = str(recovery.get("reference_query") or "").strip()
        target_asin = str(recovery.get("target_asin") or "").strip()
        guidance_kind = str(recovery.get("guidance_kind") or "none")

        lines.append("Current state summary:")
        if state_signature.get("page_type") == "search_home":
            if reference_query:
                lines.append(f"Search home. One successful reference query: {reference_query}")
            else:
                lines.append("Search home.")
        elif state_signature.get("page_type") == "search_results":
            if target_asin:
                lines.append(f"Search results; reference target product {target_asin} is visible.")
            else:
                lines.append("Search results; reference target product is visible.")
        else:
            summary_bits = []
            if target_asin:
                summary_bits.append(f"product={target_asin}")
            summary_bits.append(f"selected={self._webshop_options_text(selected_options)}")
            if remaining_options:
                summary_bits.append(
                    "remaining(any order)=" + self._webshop_options_text(remaining_options)
                )
            else:
                summary_bits.append("ready_to_buy")
            lines.append("; ".join(summary_bits) + ".")

        if guidance_kind == "option_set":
            candidate_actions = list(recovery.get("candidate_actions") or [])
            lines.append("Compatible next actions (choose one based on the current observation):")
            lines.append(", ".join(candidate_actions))
        elif guidance_kind == "search_query_set":
            # Search wording is not unique; keep reference query only in the summary.
            pass
        guessed_action = str(recovery.get("action") or "")
        if bool(recovery.get("use_action", False)) and guessed_action:
            lines.append("Candidate next action for the current state:")
            lines.append(guessed_action)
        lines.append("The current state may be on or off this path.")
        lines.append(
            "Use the path as privileged guidance, but reason from the current observation and admissible actions."
        )

    def _reset_batch_caches(self) -> None:
        self._prefix_cache_batch_id: Optional[int] = None
        self._prefix_cache: Dict[str, Dict[str, Any]] = {}

    def _ensure_prefix_cache(self, batch: DataProto) -> Dict[str, Dict[str, Any]]:
        """Build per-trajectory history caches once per teacher batch.

        Path/recovery context construction asks for the same trajectory prefix
        several times for each response row. The previous implementation scanned
        the whole flattened rollout batch on every call, which is expensive for
        ALFWorld batches with thousands of turn rows. This cache preserves the
        old semantics while making prefix lookup proportional to the trajectory
        length instead of the full batch size.
        """
        batch_id = id(batch)
        if getattr(self, "_prefix_cache_batch_id", None) == batch_id:
            return self._prefix_cache

        self._prefix_cache_batch_id = batch_id
        cache: Dict[str, Dict[str, Any]] = {}
        traj_values = batch.non_tensor_batch.get("traj_uid")
        turn_values = batch.non_tensor_batch.get("turn_step")
        action_values = batch.non_tensor_batch.get("executed_action")
        valid_values = batch.non_tensor_batch.get("is_action_valid")
        input_values = batch.non_tensor_batch.get("input")
        if traj_values is None or turn_values is None or action_values is None:
            self._prefix_cache = cache
            return cache

        for row_idx in range(len(traj_values)):
            traj_uid = str(_as_py(traj_values[row_idx]) or "")
            if not traj_uid:
                continue
            try:
                turn = int(_as_py(turn_values[row_idx]) or 0)
            except Exception:
                continue
            item = cache.setdefault(traj_uid, {"observations_by_turn": {}, "candidates": {}, "invalid_candidates": {}})
            observations_by_turn: Dict[int, str] = item["observations_by_turn"]
            if input_values is not None and turn not in observations_by_turn:
                observations_by_turn[turn] = extract_current_observation(str(_as_py(input_values[row_idx]) or ""))

            action = str(_as_py(action_values[row_idx]) or "").strip()
            if not action:
                continue
            valid = is_truthy(valid_values[row_idx], default=True) if valid_values is not None else True
            clean = is_clean_executed_action(action)
            if not valid or not clean:
                invalid_candidates: Dict[int, Dict[str, Any]] = item["invalid_candidates"]
                candidates: Dict[int, Dict[str, Any]] = item["candidates"]
                if turn not in candidates and turn not in invalid_candidates:
                    invalid_candidates[turn] = {
                        "turn": turn,
                        "action": action,
                        "reason": "non_admissible" if clean and not valid else "malformed_or_unparsed",
                    }
                continue

            candidates: Dict[int, Dict[str, Any]] = item["candidates"]
            if turn not in candidates:
                candidates[turn] = {
                    "turn": turn,
                    "action": action,
                    "valid": valid,
                    "observation": observations_by_turn.get(turn, ""),
                }

        self._prefix_cache = cache
        return cache

    def _value(self, batch: DataProto, key: str, idx: int, default: Any = None) -> Any:
        values = batch.non_tensor_batch.get(key)
        if values is None:
            return default
        try:
            return _as_py(values[idx])
        except Exception:
            return default

    def _skill_text(self, *, prompt_text: str, gamefile: Optional[str], data_source: Optional[str]) -> str:
        """Resolve teacher skill text with the same routing as native SDAR/RLSD.

        Priority matches ``build_teacher_batch`` / upstream PR #36:
          1. non-null gamefile -> gamefile substring skill (ALFWorld)
          2. data_source (including parquet ``text`` -> prompt keyword match)
          3. prompt keyword match
        """
        if self.skill_provider is None:
            return ""
        if gamefile:
            return self.skill_provider.get_privileged_info(str(gamefile))
        if data_source is not None and str(data_source):
            return self.skill_provider.get_privileged_info_from_data_source(str(data_source), prompt_text)
        return self.skill_provider.get_privileged_info_from_prompt(prompt_text)

    def _buffer_path_candidates(self, batch: DataProto, idx: int) -> List[PathCandidate]:
        gamefile = self._value(batch, "gamefile", idx)
        task_key = self._value(batch, "task_key", idx)
        trajectory_id = self._value(batch, "trajectory_id", idx)
        return self.path_buffer_index.get(gamefile, task_key, trajectory_id, max_candidates=self.path_buffer_max_candidates)

    def _filter_buffer_path_candidates(self, batch: DataProto, idx: int, candidates: List[PathCandidate], canonical_actions: List[str]) -> List[PathCandidate]:
        if not candidates:
            self._candidate_filter_metrics[idx] = {
                **self._candidate_filter_metrics.get(idx, {}),
                "path_opd/path_buffer_candidate_raw_count_mean": 0.0,
                "path_opd/path_buffer_candidate_self_drop_mean": 0.0,
                "path_opd/path_buffer_candidate_quality_drop_mean": 0.0,
                "path_opd/path_buffer_candidate_self_drop_ratio": 0.0,
                "path_opd/path_buffer_candidate_quality_drop_ratio": 0.0,
                "path_opd/path_buffer_quality_max_len_mean": 0.0,
            }
            return []

        current_traj_uid = str(self._value(batch, "traj_uid", idx, "") or "")
        canonical_len = len(canonical_actions or [])
        max_len = 0
        if self.path_buffer_max_length_ratio_to_canonical > 0 and canonical_len > 0:
            max_len = max(1, int(math.ceil(canonical_len * self.path_buffer_max_length_ratio_to_canonical)))
        if self.path_buffer_max_action_len > 0:
            max_len = min(max_len, self.path_buffer_max_action_len) if max_len > 0 else self.path_buffer_max_action_len

        kept: List[PathCandidate] = []
        self_dropped = 0
        quality_dropped = 0
        for candidate in candidates:
            is_self_success = False
            if self.dont_reprompt_on_self_success:
                is_self_success = bool(candidate.source_traj_uid and current_traj_uid and candidate.source_traj_uid == current_traj_uid)
            if is_self_success:
                self_dropped += 1
                continue
            if max_len > 0 and len(candidate.actions) > max_len:
                quality_dropped += 1
                continue
            kept.append(candidate)

        raw_count = max(1.0, float(len(candidates)))
        self._candidate_filter_metrics[idx] = {
            **self._candidate_filter_metrics.get(idx, {}),
            "path_opd/path_buffer_candidate_raw_count_mean": float(len(candidates)),
            "path_opd/path_buffer_candidate_self_drop_mean": float(self_dropped),
            "path_opd/path_buffer_candidate_quality_drop_mean": float(quality_dropped),
            "path_opd/path_buffer_candidate_self_drop_ratio": float(self_dropped) / raw_count,
            "path_opd/path_buffer_candidate_quality_drop_ratio": float(quality_dropped) / raw_count,
            "path_opd/path_buffer_quality_max_len_mean": float(max_len),
        }
        return kept

    def _path_candidates_from_sample(self, batch: DataProto, idx: int) -> List[PathCandidate]:
        gamefile = self._value(batch, "gamefile", idx)
        task_key = self._value(batch, "task_key", idx)
        trajectory_id = self._value(batch, "trajectory_id", idx)
        metadata = self._value(batch, "vmpr_metadata", idx, {}) or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        metadata_actions: List[str] = []
        metadata_source = ""
        metadata_source_traj_uid = ""
        if isinstance(metadata, dict):
            actions = extract_full_actions({"metadata": metadata})
            if actions:
                metadata_actions = actions
                metadata_source = str(metadata.get("path_source") or metadata.get("source") or "metadata")
                metadata_source_traj_uid = str(metadata.get("source_traj_uid") or metadata.get("traj_uid") or "")

        current_traj_uid = str(self._value(batch, "traj_uid", idx, "") or "")
        if self.dont_reprompt_on_self_success and metadata_source_traj_uid and current_traj_uid and metadata_source_traj_uid == current_traj_uid:
            metadata_actions = []
            self._candidate_filter_metrics[idx] = {
                **self._candidate_filter_metrics.get(idx, {}),
                "path_opd/path_metadata_self_drop_ratio": 1.0,
            }
        elif metadata_actions:
            self._candidate_filter_metrics[idx] = {
                **self._candidate_filter_metrics.get(idx, {}),
                "path_opd/path_metadata_self_drop_ratio": 0.0,
            }

        candidates: List[PathCandidate] = []
        if metadata_actions:
            candidates.append(PathCandidate(actions=metadata_actions, source=metadata_source or "metadata", source_traj_uid=metadata_source_traj_uid))

        canonical_actions = self.trajectory_index.get(gamefile, task_key, trajectory_id)
        canonical_candidate = PathCandidate(actions=canonical_actions, source="trajectory_index") if canonical_actions else None
        # WebShop keeps the oracle path in per-sample metadata (path_source=webshop_goal_oracle)
        # while TRAJECTORY_INDEX_PATH stays null. Use that oracle length for buffer quality
        # filtering so PATH_BUFFER_MAX_LENGTH_RATIO_TO_CANONICAL is not a no-op.
        quality_canonical_actions = list(canonical_actions or [])
        if not quality_canonical_actions and metadata_actions and metadata_source == "webshop_goal_oracle":
            quality_canonical_actions = list(metadata_actions)
        buffer_candidates = self._filter_buffer_path_candidates(
            batch, idx, self._buffer_path_candidates(batch, idx), quality_canonical_actions
        )

        if self.path_source_policy in {"best_aligned_success", "best_state_aligned_success"}:
            best_aligned_candidates: List[PathCandidate] = []
            if metadata_actions:
                best_aligned_candidates.append(PathCandidate(actions=metadata_actions, source=metadata_source or "metadata"))
            best_aligned_candidates.extend(buffer_candidates)
            if canonical_candidate is not None:
                best_aligned_candidates.append(canonical_candidate)
            return best_aligned_candidates or [PathCandidate(actions=[], source="missing_best_aligned_success")]

        if self.path_source_policy == "in_batch_success_only":
            if metadata_actions and metadata_source == "in_batch_success":
                return [PathCandidate(actions=metadata_actions, source="in_batch_success")]
            return [PathCandidate(actions=[], source="missing_in_batch_success")]

        if self.path_source_policy == "prefer_in_batch_success" and metadata_actions and metadata_source == "in_batch_success":
            return [PathCandidate(actions=metadata_actions, source="in_batch_success")]

        if self.path_source_policy == "buffer_success_only":
            return buffer_candidates or [PathCandidate(actions=[], source="missing_buffer_success")]

        if self.path_source_policy == "prefer_buffer_success":
            if buffer_candidates:
                return buffer_candidates
            return [canonical_candidate] if canonical_candidate is not None else [PathCandidate(actions=[], source="missing_buffer_success")]

        if self.path_source_policy == "prefer_student_success":
            ordered: List[PathCandidate] = []
            if metadata_actions and metadata_source == "in_batch_success":
                ordered.append(PathCandidate(actions=metadata_actions, source="in_batch_success"))
            ordered.extend(buffer_candidates)
            if ordered:
                return ordered
            return [canonical_candidate] if canonical_candidate is not None else [PathCandidate(actions=[], source="missing_student_success")]

        if metadata_actions:
            return [PathCandidate(actions=metadata_actions, source=metadata_source or "metadata")]

        if canonical_candidate is not None:
            return [canonical_candidate]
        return [PathCandidate(actions=[], source="missing")]

    @staticmethod
    def _longest_prefix_match(lhs: List[str], rhs: List[str]) -> int:
        count = 0
        for left, right in zip(lhs, rhs):
            if normalize_action_text(left) != normalize_action_text(right):
                break
            count += 1
        return count

    @staticmethod
    def _longest_suffix_prefix_match(history: List[str], actions: List[str]) -> int:
        max_len = min(len(history), len(actions))
        for size in range(max_len, 0, -1):
            if all(normalize_action_text(history[-size + offset]) == normalize_action_text(actions[offset]) for offset in range(size)):
                return size
        return 0

    def _select_path_candidate(self, batch: DataProto, idx: int, candidates: List[PathCandidate]) -> PathCandidate:
        if not candidates:
            return PathCandidate(actions=[], source="missing")
        real_candidates = [candidate for candidate in candidates if candidate.actions]
        if not real_candidates:
            return candidates[0]
        if len(real_candidates) == 1:
            return real_candidates[0]

        history = self._trajectory_prefix_actions(batch, idx)
        if self.path_source_policy == "best_aligned_success":
            source_priority = {
                "trajectory_index": 5,
                "in_batch_success": 4,
                "student_autonomous": 3,
                "assisted_success": 2,
                "teacher_seed": 1,
                "metadata": 0,
            }
        else:
            source_priority = {
                "in_batch_success": 5,
                "student_autonomous": 4,
                "assisted_success": 3,
                "teacher_seed": 2,
                "trajectory_index": 1,
                "metadata": 0,
            }

        def score(candidate: PathCandidate) -> Tuple[int, int, int, int, int]:
            suffix_len = self._longest_suffix_prefix_match(history, candidate.actions)
            prefix_len = self._longest_prefix_match(history, candidate.actions)
            source_score = source_priority.get(candidate.source, 0)
            # Prefer fresher student paths, then shorter successful paths as a
            # weak tie-breaker. The negative length keeps compact recoveries
            # ahead of long looping successes when alignment is equal.
            return suffix_len, prefix_len, source_score, int(candidate.insert_step), -len(candidate.actions)

        return max(real_candidates, key=score)

    def _select_state_compatible_path_candidate(
        self,
        batch: DataProto,
        idx: int,
        candidates: List[PathCandidate],
        prompt_text: str,
    ) -> tuple[PathCandidate, Optional[Dict[str, Any]], Dict[str, float]]:
        """Select a canonical/student path using the same hard state matcher as SMRC-SD.

        Prefer the highest state-match quality. When a student path and a canonical
        path are similarly compatible, keep the canonical reference. Student-success
        paths expand coverage only when they are strictly better-matched than the
        canonical candidate (or when canonical recovery cannot produce a currently
        admissible next action). If no candidate is compatible, retain the canonical
        path as non-actionable full-path context rather than selecting by history.
        """

        real_candidates = [candidate for candidate in candidates if candidate.actions]
        canonical_sources = {"trajectory_index", "webshop_goal_oracle"}
        compatible: List[tuple[tuple, PathCandidate, Dict[str, Any]]] = []
        compatible_buffer_count = 0
        for candidate in real_candidates:
            recovery = self._recovery_guess(
                batch=batch,
                idx=idx,
                prompt_text=prompt_text,
                full_actions=list(candidate.actions),
                consumed=0,
            )
            if not (recovery.get("state_match") and recovery.get("use_action") and recovery.get("match_basis") == "state"):
                continue
            if candidate.source in {"student_autonomous", "assisted_success"}:
                compatible_buffer_count += 1
            component_total = max(1, int(recovery.get("state_component_total", 1) or 1))
            component_count = int(recovery.get("state_component_match_count", 0) or 0)
            # Primary key = match quality. Canonical preference is only a tie-breaker
            # when quality is equal ("similar"), matching the WebShop/ALFWorld intent
            # that student paths expand coverage rather than displace equal matches.
            quality = (
                float(recovery.get("confidence") or 0.0),
                float(component_count) / float(component_total),
                int(recovery.get("path_start_index", 0) or 0),
            )
            canonical_tiebreak = 1 if candidate.source in canonical_sources else 0
            score = (
                quality,
                canonical_tiebreak,
                int(candidate.insert_step),
                -len(candidate.actions),
            )
            compatible.append((score, candidate, recovery))

        if compatible:
            _score, selected, recovery = max(compatible, key=lambda item: item[0])
        else:
            selected = next(
                (candidate for candidate in real_candidates if candidate.source in canonical_sources),
                real_candidates[0] if real_candidates else PathCandidate(actions=[], source="missing_best_state_aligned_success"),
            )
            recovery = None

        return selected, recovery, {
            "path_opd/path_state_compatible_candidate_count_mean": float(len(compatible)),
            "path_opd/path_state_compatible_buffer_candidate_count_mean": float(compatible_buffer_count),
            "path_opd/path_has_state_compatible_buffer_candidate_ratio": float(compatible_buffer_count > 0),
            "path_opd/path_selected_state_compatible_buffer_candidate_ratio": float(
                recovery is not None and selected.source in {"student_autonomous", "assisted_success"}
            ),
        }

    @staticmethod
    def _event_made_no_progress(event: Dict[str, Any]) -> bool:
        next_obs_norm = normalize_observation_text(str(event.get("next_observation") or ""))
        return "nothing happens" in next_obs_norm

    def _trajectory_prefix_event_groups(self, batch: DataProto, idx: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        traj_uid = self._value(batch, "traj_uid", idx)
        if not traj_uid:
            return [], [], []
        current_turn = int(self._value(batch, "turn_step", idx, 0) or 0)
        cache = self._ensure_prefix_cache(batch).get(str(traj_uid))
        if not cache:
            return [], [], []

        observations_by_turn: Dict[int, str] = cache.get("observations_by_turn", {})
        candidates: Dict[int, Dict[str, Any]] = cache.get("candidates", {})
        invalid_candidates: Dict[int, Dict[str, Any]] = cache.get("invalid_candidates", {})
        events = [dict(candidates[turn]) for turn in sorted(turn for turn in candidates if turn < current_turn)]
        for event in events:
            next_observation = observations_by_turn.get(int(event["turn"]) + 1, "")
            event["next_observation"] = next_observation
        progress_events = [event for event in events if not self._event_made_no_progress(event)]
        failed_events = [event for event in events if self._event_made_no_progress(event)]
        invalid_events = [dict(invalid_candidates[turn]) for turn in sorted(turn for turn in invalid_candidates if turn < current_turn)]
        return progress_events, failed_events, invalid_events

    @staticmethod
    def _last_known_location_from_history(events: List[Dict[str, Any]], current_observation: str) -> str:
        current_location = infer_location_from_observation(current_observation)
        if current_location != "unknown":
            return current_location

        for event in reversed(events):
            for key in ("next_observation", "observation"):
                location = infer_location_from_observation(str(event.get(key) or ""))
                if location != "unknown":
                    return location
            action = normalize_action_text(str(event.get("action") or ""))
            match = re.match(r"go to (.+)", action)
            if match:
                location = match.group(1).strip()
                if location:
                    return location
        return "unknown"

    def _trajectory_prefix_events(self, batch: DataProto, idx: int) -> List[Dict[str, Any]]:
        progress_events, _failed_events, _invalid_events = self._trajectory_prefix_event_groups(batch, idx)
        return progress_events

    def _trajectory_prefix_actions(self, batch: DataProto, idx: int) -> List[str]:
        return [event["action"] for event in self._trajectory_prefix_events(batch, idx)]

    def _last_non_no_progress_observation(self, batch: DataProto, idx: int) -> str:
        traj_uid = self._value(batch, "traj_uid", idx)
        if not traj_uid:
            return ""
        current_turn = int(self._value(batch, "turn_step", idx, 0) or 0)
        cache = self._ensure_prefix_cache(batch).get(str(traj_uid))
        if not cache:
            return ""

        best_turn = -1
        best_observation = ""
        observations_by_turn: Dict[int, str] = cache.get("observations_by_turn", {})
        for turn, observation in observations_by_turn.items():
            if turn >= current_turn or turn <= best_turn:
                continue
            if observation and "nothing happens" not in normalize_observation_text(observation):
                best_turn = turn
                best_observation = observation
        return best_observation

    @staticmethod
    def _actions_match(observed: List[str], expected: List[str]) -> bool:
        if len(observed) != len(expected):
            return False
        return [normalize_action_text(action) for action in observed] == [normalize_action_text(action) for action in expected]

    @staticmethod
    def _actions_suffix_match(observed: List[str], expected_suffix: List[str]) -> bool:
        if not expected_suffix:
            return True
        if len(observed) < len(expected_suffix):
            return False
        return [normalize_action_text(action) for action in observed[-len(expected_suffix) :]] == [normalize_action_text(action) for action in expected_suffix]

    @staticmethod
    def _actions_subsequence_match(observed: List[str], expected: List[str]) -> bool:
        expected_norm = [normalize_action_text(action) for action in expected]
        if not expected_norm:
            return True
        observed_norm = [normalize_action_text(action) for action in observed]
        cursor = 0
        for action in observed_norm:
            if cursor < len(expected_norm) and action == expected_norm[cursor]:
                cursor += 1
        return cursor == len(expected_norm)

    @staticmethod
    def _inventory_from_events(events: List[Dict[str, Any]]) -> set[str]:
        carried: set[str] = set()
        for event in events:
            action = str(event.get("action") or "")
            text = normalize_action_text(action)
            next_obs_norm = normalize_observation_text(str(event.get("next_observation") or ""))
            if "nothing happens" in next_obs_norm:
                continue
            take_match = re.match(r"take (.+?) from .+", text)
            if take_match:
                carried.add(take_match.group(1))
                continue
            move_match = re.match(r"(?:move|put) (.+?) (?:to|in|on) .+", text)
            if move_match:
                carried.discard(move_match.group(1))
                continue
            drop_match = re.match(r"drop (.+)", text)
            if drop_match:
                carried.discard(drop_match.group(1))
        return carried

    @staticmethod
    def _inventory_from_actions(actions: List[str]) -> set[str]:
        return PathPrivilegedContextProvider._inventory_from_events([{"action": action, "next_observation": ""} for action in actions])

    @staticmethod
    def _tracked_objects_from_actions(actions: List[str]) -> set[str]:
        """Return object instances whose task progress is represented by a path.

        The matcher intentionally ignores unrelated exploration state. Only
        objects manipulated by the reference path participate in progress
        comparison, which avoids rejecting a valid recovery because the student
        opened an irrelevant container while off path.
        """
        tracked: set[str] = set()
        patterns = [
            r"take (.+?) from .+",
            r"(?:move|put) (.+?) (?:to|in|on) .+",
            r"(?:clean|cool|heat|slice) (.+?) with .+",
            r"drop (.+)",
        ]
        for action in actions:
            text = normalize_action_text(action)
            for pattern in patterns:
                match = re.fullmatch(pattern, text)
                if match:
                    tracked.add(match.group(1).strip())
                    break
        return tracked

    @staticmethod
    def _state_location(events: List[Dict[str, Any]], observation: str) -> str:
        # Use the same transition-based location rule for both the current and
        # reference states. Observations can mention a different receptacle
        # after actions such as `open safe 1` or `use desklamp 1`, even though
        # the latest successful navigation action was to another receptacle.
        for event in reversed(events):
            action = normalize_action_text(str(event.get("action") or ""))
            match = re.fullmatch(r"go to (.+)", action)
            if match:
                return match.group(1).strip()
        location = normalize_action_text(infer_location_from_observation(observation))
        if location and location != "unknown":
            return location
        return "unknown"

    @classmethod
    def _execution_state_signature(
        cls,
        events: List[Dict[str, Any]],
        observation: str,
        tracked_objects: set[str],
        object_match_mode: str = "exact",
    ) -> Dict[str, Any]:
        """Build a compact, deterministic ALFWorld execution-state signature.

        This is deliberately narrower than the S-series natural-language state
        summary. It uses only successful action transitions plus the current
        observation and exposes every compared field for auditability.
        """
        inventory: set[str] = set()
        object_locations: Dict[str, str] = {}
        object_properties: set[str] = set()
        activated_tools: set[str] = set()
        semantic_tracked_objects = (
            {cls._semantic_object_name(obj) for obj in tracked_objects}
            if object_match_mode == "semantic_instance"
            else set()
        )

        def is_tracked(obj: str) -> bool:
            return obj in tracked_objects or (
                object_match_mode == "semantic_instance"
                and cls._semantic_object_name(obj) in semantic_tracked_objects
            )

        for event in events:
            next_obs_norm = normalize_observation_text(str(event.get("next_observation") or ""))
            if "nothing happens" in next_obs_norm:
                continue
            action = normalize_action_text(str(event.get("action") or ""))
            take_match = re.fullmatch(r"take (.+?) from (.+)", action)
            if take_match:
                obj = take_match.group(1).strip()
                inventory.add(obj)
                if is_tracked(obj):
                    object_locations[obj] = "inventory"
                continue
            move_match = re.fullmatch(r"(?:move|put) (.+?) (?:to|in|on) (.+)", action)
            if move_match:
                obj, destination = move_match.group(1).strip(), move_match.group(2).strip()
                inventory.discard(obj)
                if is_tracked(obj):
                    object_locations[obj] = destination
                continue
            transform_match = re.fullmatch(r"(clean|cool|heat|slice) (.+?) with (.+)", action)
            if transform_match:
                operation, obj = transform_match.group(1), transform_match.group(2).strip()
                if is_tracked(obj):
                    # ALFWorld temperature is mutually exclusive: heating an
                    # object removes isCool, while cooling it removes isHot.
                    # Keep the reduced execution state consistent with those
                    # environment transitions instead of accumulating both.
                    if operation == "heat":
                        object_properties.discard(f"cool:{obj}")
                    elif operation == "cool":
                        object_properties.discard(f"heat:{obj}")
                    object_properties.add(f"{operation}:{obj}")
                continue
            drop_match = re.fullmatch(r"drop (.+)", action)
            if drop_match:
                obj = drop_match.group(1).strip()
                inventory.discard(obj)
                if is_tracked(obj):
                    object_locations[obj] = "dropped"
                continue
            use_match = re.fullmatch(r"use (.+)", action)
            if use_match:
                activated_tools.add(use_match.group(1).strip())

        return {
            "location": cls._state_location(events, observation),
            "inventory": tuple(sorted(inventory)),
            "object_locations": tuple(sorted(object_locations.items())),
            "object_properties": tuple(sorted(object_properties)),
            "activated_tools": tuple(sorted(activated_tools)),
        }

    @staticmethod
    def _state_signature_text(signature: Dict[str, Any]) -> str:
        """Render a compact teacher-facing state summary.

        ALFWorld keeps the historical one-line summary used by the successful
        MatchOnly ablations, except `activated_tools` is matcher-internal only
        and is never rendered into the teacher prompt.
        """
        if "page_type" in signature:
            normalized = normalize_webshop_state(signature)
            return "; ".join(
                [
                    f"page_type={normalized['page_type']}",
                    f"asin={normalized['asin'] or '(none)'}",
                    f"selected_options={normalized['options'] or {}}",
                ]
            )
        return "; ".join(
            [
                f"location={signature.get('location', 'unknown')}",
                f"inventory={list(signature.get('inventory', ())) or []}",
                f"object_locations={dict(signature.get('object_locations', ())) or {}}",
                f"object_properties={list(signature.get('object_properties', ())) or []}",
            ]
        )

    @staticmethod
    def _semantic_object_name(obj: str) -> str:
        """Remove only an ALFWorld pickup-object instance suffix.

        This helper is applied exclusively to object arguments of manipulation
        actions. Receptacle, location, and tool identifiers stay exact because
        they constrain navigation and action admissibility.
        """

        normalized = normalize_action_text(obj)
        return re.sub(r"\s+\d+$", "", normalized).strip()

    @classmethod
    def _semantic_inventory_counter(cls, signature: Dict[str, Any]) -> Counter:
        return Counter(cls._semantic_object_name(obj) for obj in signature.get("inventory", ()))

    @classmethod
    def _semantic_location_counter(cls, signature: Dict[str, Any]) -> Counter:
        return Counter(
            (cls._semantic_object_name(obj), str(location))
            for obj, location in signature.get("object_locations", ())
        )

    @classmethod
    def _semantic_property_counter(cls, signature: Dict[str, Any]) -> Counter:
        facts = Counter()
        for fact in signature.get("object_properties", ()):
            operation, separator, obj = str(fact).partition(":")
            if separator:
                facts[(operation, cls._semantic_object_name(obj))] += 1
            else:
                facts[(str(fact), "")] += 1
        return facts

    @classmethod
    def _semantic_activated_tool_counter(cls, signature: Dict[str, Any]) -> Counter:
        return Counter(cls._semantic_object_name(tool) for tool in signature.get("activated_tools", ()))

    @classmethod
    def _state_component_compatibility(
        cls,
        current: Dict[str, Any],
        reference: Dict[str, Any],
        object_match_mode: str = "exact",
    ) -> Dict[str, bool]:
        """Compare execution states using reference-satisfying compatibility.

        Location and inventory are exact because they directly constrain the
        next action.  Progress facts are monotonic requirements: a student may
        already have completed additional task-object placements or transforms
        without invalidating a shorter canonical continuation.
        """

        if object_match_mode == "semantic_instance":
            current_locations = cls._semantic_location_counter(current)
            reference_locations = cls._semantic_location_counter(reference)
            current_properties = cls._semantic_property_counter(current)
            reference_properties = cls._semantic_property_counter(reference)
            inventory_match = cls._semantic_inventory_counter(current) == cls._semantic_inventory_counter(reference)
            # Semantic-instance matching represents the task-object state as a
            # multiset quotient. Keep the phase exact: relaxing both instance
            # identity and monotonic progress would let a completed transform
            # or placement align back to an earlier reference state.
            object_locations_match = current_locations == reference_locations
            object_properties_match = current_properties == reference_properties
            activated_tools_match = cls._semantic_activated_tool_counter(current) == cls._semantic_activated_tool_counter(reference)
        else:
            current_locations = dict(current.get("object_locations", ()))
            reference_locations = dict(reference.get("object_locations", ()))
            inventory_match = current.get("inventory") == reference.get("inventory")
            object_locations_match = all(current_locations.get(obj) == location for obj, location in reference_locations.items())
            object_properties_match = set(reference.get("object_properties", ())).issubset(set(current.get("object_properties", ())))
            activated_tools_match = set(reference.get("activated_tools", ())).issubset(set(current.get("activated_tools", ())))
        return {
            "location": current.get("location") == reference.get("location"),
            "inventory": inventory_match,
            "object_locations": object_locations_match,
            "object_properties": object_properties_match,
            "activated_tools": activated_tools_match,
        }

    @classmethod
    def _semantic_action_key(cls, action: str) -> Tuple[str, ...]:
        """Return an action key that abstracts only manipulated-object IDs."""

        text = normalize_action_text(action)
        take_match = re.fullmatch(r"take (.+?) from (.+)", text)
        if take_match:
            return ("take", cls._semantic_object_name(take_match.group(1)), take_match.group(2).strip())
        move_match = re.fullmatch(r"(move|put) (.+?) (to|in|on) (.+)", text)
        if move_match:
            return (
                move_match.group(1),
                cls._semantic_object_name(move_match.group(2)),
                move_match.group(3),
                cls._semantic_object_name(move_match.group(4)),
            )
        transform_match = re.fullmatch(r"(clean|cool|heat|slice) (.+?) with (.+)", text)
        if transform_match:
            return (
                transform_match.group(1),
                cls._semantic_object_name(transform_match.group(2)),
                transform_match.group(3).strip(),
            )
        drop_match = re.fullmatch(r"drop (.+)", text)
        if drop_match:
            return ("drop", cls._semantic_object_name(drop_match.group(1)))
        use_match = re.fullmatch(r"use (.+)", text)
        if use_match:
            return ("use", cls._semantic_object_name(use_match.group(1)))
        return ("exact", text)

    @classmethod
    def _semantic_action_location_match(
        cls,
        current: Dict[str, Any],
        reference: Dict[str, Any],
        grounded_action: str,
        reference_action: str,
    ) -> bool:
        """Allow role-equivalent goal receptacles/tools to rename their location."""

        grounded_text = normalize_action_text(grounded_action)
        reference_text = normalize_action_text(reference_action)
        grounded_move = re.fullmatch(r"(?:move|put) .+? (?:to|in|on) (.+)", grounded_text)
        reference_move = re.fullmatch(r"(?:move|put) .+? (?:to|in|on) (.+)", reference_text)
        if grounded_move and reference_move:
            grounded_destination = grounded_move.group(1).strip()
            reference_destination = reference_move.group(1).strip()
            return (
                cls._semantic_object_name(grounded_destination) == cls._semantic_object_name(reference_destination)
                and current.get("location") == grounded_destination
                and reference.get("location") == reference_destination
            )
        if grounded_text.startswith("use ") and reference_text.startswith("use "):
            return (
                grounded_text != reference_text
                and cls._semantic_action_key(grounded_action) == cls._semantic_action_key(reference_action)
                and current.get("location") not in {None, "", "unknown"}
                and reference.get("location") not in {None, "", "unknown"}
            )
        return False

    @classmethod
    def _ground_reference_action(
        cls,
        reference_action: str,
        admissible_actions: List[str],
        object_match_mode: str,
    ) -> List[str]:
        reference_norm = normalize_action_text(reference_action)
        exact = [action for action in admissible_actions if normalize_action_text(action) == reference_norm]
        if exact or object_match_mode != "semantic_instance":
            return exact
        reference_key = cls._semantic_action_key(reference_action)
        if reference_key[0] == "exact":
            return []
        return [action for action in admissible_actions if cls._semantic_action_key(action) == reference_key]

    @staticmethod
    def _visited_locations_from_actions(actions: List[str]) -> List[str]:
        visited: List[str] = []
        seen: set[str] = set()
        for action in actions:
            text = normalize_action_text(action)
            match = re.match(r"go to (.+)", text)
            if not match:
                continue
            location = match.group(1).strip()
            if location and location not in seen:
                visited.append(location)
                seen.add(location)
        return visited

    @staticmethod
    def _repeated_actions(actions: List[str]) -> List[str]:
        counts = Counter(normalize_action_text(action) for action in actions if action)
        repeated = [action for action, count in counts.most_common() if action and count > 1]
        return repeated[:8]

    @staticmethod
    def _negative_constraints_from_history(actions: List[str], current_observation: str) -> List[str]:
        constraints: List[str] = []
        repeated = PathPrivilegedContextProvider._repeated_actions(actions)
        if repeated:
            constraints.append("avoid repeating actions already tried many times: " + " -> ".join(repeated[:4]))
        if actions and len(actions) >= 2 and normalize_action_text(actions[-1]) == normalize_action_text(actions[-2]):
            constraints.append(f"last action repeated without progress: {actions[-1]}")
        obs_norm = normalize_observation_text(current_observation)
        if "nothing happens" in obs_norm and actions:
            constraints.append(f"latest action produced no progress: {actions[-1]}")
        if "you see nothing" in obs_norm:
            constraints.append("current container or surface appears empty")
        return constraints[:6]

    @staticmethod
    def _state_observation_flags(current_observation: str) -> List[str]:
        obs_norm = normalize_observation_text(current_observation)
        flags: List[str] = []
        if "nothing happens" in obs_norm:
            flags.append("last_action_had_no_effect")
        if "you see nothing" in obs_norm:
            flags.append("no_visible_object")
        if "closed" in obs_norm:
            flags.append("closed_container")
        if "open" in obs_norm:
            flags.append("open_container")
        return flags[:4]

    def _state_history_block(self, batch: DataProto, idx: int, prompt_text: str) -> tuple[str, Dict[str, float]]:
        current_observation = extract_current_observation(prompt_text)
        history_events, failed_events, invalid_events = self._trajectory_prefix_event_groups(batch, idx)
        current_location = self._last_known_location_from_history(sorted(history_events + failed_events, key=lambda event: int(event.get("turn", 0) or 0)), current_observation)
        history_actions = [str(event.get("action") or "") for event in history_events if event.get("action")]
        recent_actions = history_actions[-8:]
        inventory = sorted(self._inventory_from_events(history_events))
        visited_locations = self._visited_locations_from_actions(history_actions)
        repeated_actions = self._repeated_actions(history_actions)
        sampled_action = str(self._value(batch, "executed_action", idx, "") or "").strip()
        sampled_norm = normalize_action_text(sampled_action)
        action_counts = Counter(normalize_action_text(action) for action in history_actions if action)
        sampled_action_seen = bool(sampled_norm and sampled_norm in action_counts)
        sampled_action_repeats_previous = bool(sampled_norm and history_actions and normalize_action_text(history_actions[-1]) == sampled_norm)
        no_progress_action_history = [
            f"turn {event.get('turn')}: {compact_action_text(event.get('action'))} -> nothing happened"
            for event in failed_events[-5:]
            if event.get("action")
        ]
        last_invalid_action = ""
        if failed_events:
            latest_failed = failed_events[-1]
            if int(latest_failed.get("turn", -999) or -999) == int(self._value(batch, "turn_step", idx, 0) or 0) - 1:
                last_invalid_action = f"turn {latest_failed.get('turn')}: {compact_action_text(latest_failed.get('action'))} -> nothing happened"
        if not last_invalid_action and invalid_events:
            latest_invalid = invalid_events[-1]
            if int(latest_invalid.get("turn", -999) or -999) == int(self._value(batch, "turn_step", idx, 0) or 0) - 1:
                reason = str(latest_invalid.get("reason") or "invalid")
                last_invalid_action = f"turn {latest_invalid.get('turn')}: {compact_action_text(latest_invalid.get('action'))} -> {reason}"
        reminders: List[str] = []
        if repeated_actions:
            reminders.append("Avoid unproductive retries listed in repeated_action_history.")
        if last_invalid_action:
            reminders.append("The invalid_action failed in the immediately previous context; retry only if the state has changed.")

        lines = ["[History Summary]"]
        lines.append("This is a concise execution-history summary for the current ALFWorld episode.")
        lines.append(f"current_location: {current_location}")
        lines.append(f"inventory: {inventory if inventory else []}")
        lines.append(f"visited_locations: {visited_locations if visited_locations else []}")
        lines.append(f"recent_action_history: {recent_actions if recent_actions else []}")
        if repeated_actions:
            lines.append(f"repeated_action_history: {repeated_actions}")
        if last_invalid_action:
            lines.append(f"invalid_action: {last_invalid_action}")
        if reminders:
            lines.append(f"state_reminder: {' '.join(reminders)}")
        lines.append("[/History Summary]")

        return "\n".join(lines), {
            "path_opd/has_state_history_ratio": 1.0,
            "path_opd/state_history_actions_mean": float(len(history_actions)),
            "path_opd/state_history_recent_actions_mean": float(len(recent_actions)),
            "path_opd/state_history_failed_actions_mean": float(len(failed_events)),
            "path_opd/state_history_recent_failed_actions_mean": float(len(no_progress_action_history)),
            "path_opd/state_history_invalid_actions_mean": float(len(invalid_events)),
            "path_opd/state_history_inventory_items_mean": float(len(inventory)),
            "path_opd/state_history_visited_locations_mean": float(len(visited_locations)),
            "path_opd/state_history_repeated_actions_mean": float(len(repeated_actions)),
            "path_opd/state_history_reminders_mean": float(len(reminders)),
            "path_opd/state_history_sampled_action_seen_ratio": float(sampled_action_seen),
            "path_opd/state_history_sampled_action_repeats_previous_ratio": float(sampled_action_repeats_previous),
        }

    def _alignment_state(self, batch: DataProto, idx: int, full_actions: List[str]) -> tuple[str, int, List[str], List[str]]:
        turn_step = int(self._value(batch, "turn_step", idx, 0) or 0)
        vmpr_prefix_len = int(self._value(batch, "vmpr_prefix_len", idx, 0) or 0)
        is_prefix_start = bool(self._value(batch, "vmpr_is_prefix_start", idx, False))

        if self.alignment_mode == "legacy":
            consumed = max(0, min(len(full_actions), vmpr_prefix_len + turn_step))
            return "legacy", consumed, full_actions[:consumed], full_actions[consumed : consumed + self.max_remaining_actions]

        if self.alignment_mode != "dynamic":
            return "full_start", 0, [], full_actions[: self.max_remaining_actions]

        observed_suffix = self._trajectory_prefix_actions(batch, idx)
        if turn_step == 0 and full_actions:
            return "aligned_full_start", 0, [], full_actions[: self.max_remaining_actions]
        if is_prefix_start:
            consumed = max(0, min(len(full_actions), vmpr_prefix_len + turn_step))
            expected_suffix = full_actions[vmpr_prefix_len:consumed]
            if self._actions_match(observed_suffix, expected_suffix):
                return "aligned_replay", consumed, full_actions[:consumed], full_actions[consumed : consumed + self.max_remaining_actions]
            return "deviated_replay", 0, [], full_actions[: self.max_remaining_actions]

        consumed = max(0, min(len(full_actions), len(observed_suffix)))
        expected_prefix = full_actions[:consumed]
        if self._actions_match(observed_suffix, expected_prefix):
            return "aligned_full_start", consumed, full_actions[:consumed], full_actions[consumed : consumed + self.max_remaining_actions]
        if self.enable_history_suffix_alignment:
            suffix_consumed = self._longest_suffix_prefix_match(observed_suffix, full_actions)
            if suffix_consumed > 0:
                return "aligned_history_suffix", suffix_consumed, full_actions[:suffix_consumed], full_actions[suffix_consumed : suffix_consumed + self.max_remaining_actions]

        return "deviated_full_start", 0, [], full_actions[: self.max_remaining_actions]

    def _webshop_goal_progress_recovery_guess(
        self,
        *,
        batch: DataProto,
        idx: int,
        prompt_text: str,
        full_actions: List[str],
    ) -> Dict[str, Any]:
        """Match WebShop progress against the goal, not one serialized path."""
        admissible_actions = extract_admissible_actions(prompt_text)
        if not admissible_actions:
            return {
                "route": "missing_admissible_actions",
                "action": "",
                "continuation": [],
                "confidence": 0.0,
                "use_action": False,
                "state_match": False,
                "match_basis": "state",
            }

        current_raw = self._value(batch, "webshop_state", idx, {}) or {}
        current_state = normalize_webshop_state(current_raw if isinstance(current_raw, dict) else {})
        if current_state.get("page_type") == "unknown":
            return {
                "route": "missing_webshop_state",
                "action": "",
                "continuation": [],
                "confidence": 0.0,
                "use_action": False,
                "state_match": False,
                "match_basis": "state",
            }

        metadata = self._value(batch, "vmpr_metadata", idx, {}) or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        metadata = metadata if isinstance(metadata, dict) else {}
        target_asin = str(metadata.get("webshop_target_asin") or "").strip()
        target_options = dict(metadata.get("webshop_target_options") or {})
        reference_states = list(metadata.get("webshop_reference_states") or [])
        if not target_asin:
            target_asin = next(
                (
                    str(state.get("asin") or "").strip()
                    for state in reference_states
                    if isinstance(state, dict) and state.get("asin")
                ),
                "",
            )
        if not target_options:
            option_states = [
                dict(state.get("options") or {})
                for state in reference_states
                if isinstance(state, dict) and state.get("options")
            ]
            target_options = max(option_states, key=len) if option_states else {}
        reference_query = webshop_action_argument(full_actions[0]) if full_actions else ""
        guidance = webshop_goal_progress_guidance(
            current_state,
            target_asin=target_asin,
            target_options=target_options,
            available_actions=admissible_actions,
            reference_query=reference_query,
        )
        component_match = dict(guidance.get("state_component_match") or {})
        selected_options = dict(guidance.get("selected_options") or {})
        remaining_options = dict(guidance.get("remaining_options") or {})
        page_type = current_state.get("page_type")
        if page_type == "search_home":
            path_start_index = 0
        elif page_type == "search_results":
            path_start_index = 1
        else:
            path_start_index = 2 + len(selected_options)
        result = {
            **guidance,
            "continuation": list(guidance.get("candidate_actions") or []),
            "path_start_index": path_start_index,
            "trace_history_len": path_start_index,
            "inventory_match": True,
            "state_component_match_count": sum(component_match.values()),
            "state_component_total": len(component_match) or 3,
            "state_component_match": component_match,
            "state_signature": current_state,
            "match_basis": "state",
            "prompt_mode": "webshop_goal_progress",
            "symbolic_reference": True,
            "used_last_progress_observation": False,
        }
        sampled_action = normalize_action_text(self._value(batch, "executed_action", idx, "") or "")
        candidate_action = str(result.get("action") or "")
        if candidate_action:
            result["candidate_matches_student"] = normalize_action_text(candidate_action) == sampled_action
        if (
            (self.recovery_correction_only or self.recovery_skip_if_action_matches_student)
            and bool(result.get("use_action", False))
            and result.get("candidate_matches_student", False)
        ):
            result.update(
                {
                    "route": "webshop_state_candidate_matches_student",
                    "action": "",
                    "confidence": 0.0,
                    "use_action": False,
                }
            )
        return result

    def _webshop_recovery_guess(
        self,
        *,
        batch: DataProto,
        idx: int,
        prompt_text: str,
        full_actions: List[str],
    ) -> Dict[str, Any]:
        """Match WebShop's probe-validated progress state to an oracle suffix."""
        if self.recovery_webshop_match_mode == "goal_progress":
            return self._webshop_goal_progress_recovery_guess(
                batch=batch,
                idx=idx,
                prompt_text=prompt_text,
                full_actions=full_actions,
            )
        admissible_actions = extract_admissible_actions(prompt_text)
        if not admissible_actions:
            return {
                "route": "missing_admissible_actions",
                "action": "",
                "continuation": [],
                "confidence": 0.0,
                "use_action": False,
                "state_match": False,
                "match_basis": "state",
            }

        current_raw = self._value(batch, "webshop_state", idx, {}) or {}
        current_state = normalize_webshop_state(current_raw if isinstance(current_raw, dict) else {})
        if current_state.get("page_type") == "unknown":
            return {
                "route": "missing_webshop_state",
                "action": "",
                "continuation": [],
                "confidence": 0.0,
                "use_action": False,
                "state_match": False,
                "match_basis": "state",
            }

        metadata = self._value(batch, "vmpr_metadata", idx, {}) or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        reference_states = list(metadata.get("webshop_reference_states") or []) if isinstance(metadata, dict) else []
        if len(reference_states) != len(full_actions):
            return {
                "route": "missing_webshop_reference_states",
                "action": "",
                "continuation": [],
                "confidence": 0.0,
                "use_action": False,
                "state_match": False,
                "match_basis": "state",
                "state_signature": current_state,
                "symbolic_reference": True,
            }

        sampled_action = normalize_action_text(self._value(batch, "executed_action", idx, "") or "")
        state_matches: List[Tuple[int, str, Dict[str, Any], Dict[str, bool]]] = []
        scored_states: List[Tuple[int, int, str, Dict[str, Any], Dict[str, bool]]] = []
        for step_index, (next_action, reference_raw) in enumerate(zip(full_actions, reference_states)):
            if not webshop_action_is_admissible(next_action, admissible_actions):
                continue
            if self.recovery_skip_if_action_matches_student and normalize_action_text(next_action) == sampled_action:
                continue
            reference_state = normalize_webshop_state(reference_raw if isinstance(reference_raw, dict) else {})
            component_match = webshop_state_component_match(current_state, reference_state)
            component_count = sum(component_match.values())
            scored_states.append((component_count, step_index, next_action, reference_state, component_match))
            if webshop_progress_state_key(current_state) == webshop_progress_state_key(reference_state):
                state_matches.append((step_index, next_action, reference_state, component_match))

        if state_matches:
            best_step_index, best_next_action, reference_state, component_match = max(state_matches, key=lambda item: item[0])
            result = {
                "route": "webshop_progress_state_compatible",
                "action": best_next_action,
                "continuation": full_actions[best_step_index : best_step_index + self.max_remaining_actions],
                "confidence": 1.0,
                "use_action": True,
                "path_start_index": best_step_index,
                "trace_history_len": best_step_index,
                "inventory_match": True,
                "state_match": True,
                "state_component_match_count": len(component_match),
                "state_component_total": len(component_match),
                "state_component_match": component_match,
                "state_signature": current_state,
                "trace_state_signature": reference_state,
                "match_basis": "state",
                "symbolic_reference": True,
                "used_last_progress_observation": False,
            }
            if self.recovery_correction_only and normalize_action_text(best_next_action) == sampled_action:
                result.update(
                    {
                        "route": "webshop_state_candidate_matches_student",
                        "action": "",
                        "confidence": 0.0,
                        "use_action": False,
                        "candidate_matches_student": True,
                    }
                )
            return result

        if scored_states:
            component_count, best_step_index, _next_action, reference_state, component_match = max(
                scored_states,
                key=lambda item: (item[0], item[1]),
            )
            return {
                "route": "webshop_progress_state_mismatch",
                "action": "",
                "continuation": [],
                "confidence": 0.0,
                "use_action": False,
                "path_start_index": best_step_index,
                "trace_history_len": best_step_index,
                "inventory_match": True,
                "state_match": False,
                "state_component_match_count": component_count,
                "state_component_total": len(component_match),
                "state_component_match": component_match,
                "state_signature": current_state,
                "trace_state_signature": reference_state,
                "match_basis": "state",
                "symbolic_reference": True,
                "used_last_progress_observation": False,
            }

        return {
            "route": "no_webshop_oracle_action_currently_admissible",
            "action": "",
            "continuation": [],
            "confidence": 0.0,
            "use_action": False,
            "state_match": False,
            "state_component_match_count": 0,
            "state_component_total": 3,
            "state_component_match": {},
            "state_signature": current_state,
            "match_basis": "state",
            "symbolic_reference": True,
            "used_last_progress_observation": False,
        }

    def _recovery_guess(
        self,
        *,
        batch: DataProto,
        idx: int,
        prompt_text: str,
        full_actions: List[str],
        consumed: int,
    ) -> Dict[str, Any]:
        used_last_progress_observation = False
        admissible_actions = extract_admissible_actions(prompt_text)
        admissible_by_norm = {normalize_action_text(action): action for action in admissible_actions}
        if not full_actions:
            return {"route": "missing_gt_path", "action": "", "continuation": [], "confidence": 0.0, "use_action": False, "used_last_progress_observation": False}
        if not admissible_by_norm:
            return {"route": "missing_admissible_actions", "action": "", "continuation": [], "confidence": 0.0, "use_action": False, "used_last_progress_observation": False}
        if (
            self.enable_walkthrough_trace_recovery
            and self.recovery_match_mode == "state"
            and getattr(self, "recovery_state_domain", "alfworld") == "webshop"
        ):
            return self._webshop_recovery_guess(
                batch=batch,
                idx=idx,
                prompt_text=prompt_text,
                full_actions=full_actions,
            )

        prompt_observation = extract_current_observation(prompt_text)
        prompt_obs = normalize_observation_text(prompt_observation)
        state_observation = prompt_observation
        if self.recovery_use_last_progress_observation_on_no_progress and "nothing happens" in prompt_obs:
            last_progress_observation = self._last_non_no_progress_observation(batch, idx)
            if last_progress_observation:
                prompt_obs = normalize_observation_text(last_progress_observation)
                state_observation = last_progress_observation
                used_last_progress_observation = True
        sampled_action = normalize_action_text(self._value(batch, "executed_action", idx, "") or "")
        student_history = self._trajectory_prefix_actions(batch, idx)
        student_history_norm = [normalize_action_text(action) for action in student_history]
        student_events = self._trajectory_prefix_events(batch, idx)
        gamefile = self._value(batch, "gamefile", idx)
        if self.enable_walkthrough_trace_recovery and self.recovery_match_mode == "state":
            object_match_mode = getattr(self, "recovery_alfworld_object_match_mode", "exact")
            if "nothing happens" in prompt_obs:
                last_progress_observation = self._last_non_no_progress_observation(batch, idx)
                if last_progress_observation:
                    state_observation = last_progress_observation
                    used_last_progress_observation = True
            tracked_objects = self._tracked_objects_from_actions(full_actions)
            current_signature = self._execution_state_signature(
                student_events,
                state_observation,
                tracked_objects,
                object_match_mode,
            )
            state_matches: List[Tuple[int, str, str, Dict[str, Any], int]] = []
            scored_states: List[Tuple[int, int, str, Dict[str, Any]]] = []
            # Structured-state recovery needs only the successful action prefix.
            # Replaying a TextWorld environment here produced the same signature
            # while adding one environment construction per unseen (game, path)
            # and retaining an unbounded trace cache.  Build the canonical state
            # symbolically instead; history-mode recovery still uses TextWorld
            # observations below.
            for step_index, next_action in enumerate(full_actions):
                grounded_actions = self._ground_reference_action(
                    next_action,
                    admissible_actions,
                    object_match_mode,
                )
                if not grounded_actions:
                    continue
                grounded_action = grounded_actions[0]
                if self.recovery_skip_if_action_matches_student and normalize_action_text(grounded_action) == sampled_action:
                    continue
                trace_events = [{"action": action, "next_observation": ""} for action in full_actions[:step_index]]
                # Before the first action ALFWorld starts in the canonical
                # middle-of-room navigation state.  Later locations are
                # determined by the latest successful `go to` prefix action.
                trace_observation = "You are in the middle of a room." if step_index == 0 else ""
                trace_signature = self._execution_state_signature(
                    trace_events,
                    trace_observation,
                    tracked_objects,
                    object_match_mode,
                )
                component_match = self._state_component_compatibility(
                    current_signature,
                    trace_signature,
                    object_match_mode,
                )
                if (
                    object_match_mode == "semantic_instance"
                    and not component_match["location"]
                    and self._semantic_action_location_match(
                        current_signature,
                        trace_signature,
                        grounded_action,
                        next_action,
                    )
                ):
                    component_match["location"] = True
                component_matches = sum(component_match.values())
                scored_states.append((component_matches, step_index, next_action, trace_signature))
                if all(component_match.values()):
                    state_matches.append((step_index, grounded_action, next_action, trace_signature, len(grounded_actions)))

            if state_matches:
                best_step_index, best_next_action, reference_next_action, trace_signature, grounded_action_count = max(
                    state_matches,
                    key=lambda item: item[0],
                )
                result = {
                    "route": "walkthrough_state_compatible",
                    "action": best_next_action,
                    "reference_action": reference_next_action,
                    "continuation": full_actions[best_step_index : best_step_index + self.max_remaining_actions],
                    "confidence": 1.0,
                    "use_action": True,
                    "path_start_index": best_step_index,
                    "trace_history_len": best_step_index,
                    "inventory_match": current_signature["inventory"] == trace_signature["inventory"],
                    "state_match": True,
                    "state_component_match_count": 5,
                    "state_component_match": {key: True for key in ("location", "inventory", "object_locations", "object_properties", "activated_tools")},
                    "state_signature": current_signature,
                    "trace_state_signature": trace_signature,
                    "match_basis": "state",
                    "symbolic_reference": True,
                    "used_last_progress_observation": used_last_progress_observation,
                    "object_match_mode": object_match_mode,
                    "semantic_action_grounded": normalize_action_text(best_next_action) != normalize_action_text(reference_next_action),
                    "grounded_action_count": grounded_action_count,
                }
                if self.recovery_correction_only and normalize_action_text(best_next_action) == sampled_action:
                    result.update(
                        {
                            "route": "walkthrough_state_candidate_matches_student",
                            "action": "",
                            "confidence": 0.0,
                            "use_action": False,
                            "candidate_matches_student": True,
                        }
                    )
                return result

            if scored_states:
                best_component_count, best_step_index, _best_next_action, trace_signature = max(scored_states, key=lambda item: (item[0], item[1]))
                best_component_match = self._state_component_compatibility(
                    current_signature,
                    trace_signature,
                    object_match_mode,
                )
                return {
                    "route": "walkthrough_state_mismatch",
                    "action": "",
                    "continuation": [],
                    "confidence": 0.0,
                    "use_action": False,
                    "path_start_index": best_step_index,
                    "trace_history_len": best_step_index,
                    "inventory_match": best_component_match["inventory"],
                    "state_match": False,
                    "state_component_match_count": best_component_count,
                    "state_component_match": best_component_match,
                    "state_signature": current_signature,
                    "trace_state_signature": trace_signature,
                    "match_basis": "state",
                    "symbolic_reference": True,
                    "used_last_progress_observation": used_last_progress_observation,
                    "object_match_mode": object_match_mode,
                }
            return {
                "route": "no_walkthrough_action_currently_admissible",
                "action": "",
                "continuation": [],
                "confidence": 0.0,
                "use_action": False,
                "state_match": False,
                "state_component_match_count": 0,
                "state_component_match": {},
                "state_signature": current_signature,
                "match_basis": "state",
                "symbolic_reference": True,
                "used_last_progress_observation": used_last_progress_observation,
                "object_match_mode": object_match_mode,
            }

        trace = self.trace_cache.get(gamefile, full_actions) if self.enable_walkthrough_trace_recovery else []
        if trace and prompt_obs:
            matches: List[WalkthroughState] = []
            for state in trace:
                state_obs_norm = state.observation_norm or normalize_observation_text(state.observation)
                if state_obs_norm != prompt_obs:
                    continue
                next_norm = state.next_action_norm or normalize_action_text(state.next_action)
                if next_norm in admissible_by_norm and (not self.recovery_skip_if_action_matches_student or next_norm != sampled_action):
                    matches.append(state)
            if matches:
                def score(state: WalkthroughState) -> Tuple[int, int, int]:
                    trace_history = list(state.action_history)
                    trace_history_norm = state.action_history_norm or [normalize_action_text(action) for action in trace_history]
                    suffix_score = len(trace_history) if trace_history_norm and len(student_history_norm) >= len(trace_history_norm) and student_history_norm[-len(trace_history_norm) :] == trace_history_norm else (-1 if trace_history_norm else 0)
                    cursor = 0
                    for action in student_history_norm:
                        if cursor < len(trace_history_norm) and action == trace_history_norm[cursor]:
                            cursor += 1
                    subseq_score = len(trace_history) if cursor == len(trace_history_norm) else -1
                    prefix_score = 0
                    for lhs, rhs in zip(student_history_norm, trace_history_norm):
                        if lhs != rhs:
                            break
                        prefix_score += 1
                    return suffix_score, subseq_score, prefix_score

                best = max(matches, key=lambda state: (*score(state), state.step_index))
                trace_history = list(best.action_history)
                trace_history_norm = best.action_history_norm or [normalize_action_text(action) for action in trace_history]
                if not trace_history and int(self._value(batch, "turn_step", idx, 0) or 0) == 0:
                    route = "walkthrough_trace_obs_match_initial"
                    confidence = 0.0
                elif trace_history_norm and len(student_history_norm) >= len(trace_history_norm) and student_history_norm[-len(trace_history_norm) :] == trace_history_norm:
                    route = "walkthrough_trace_obs_match_history_suffix"
                    confidence = 1.0
                elif self._actions_subsequence_match(student_history, trace_history):
                    route = "walkthrough_trace_obs_match_history_subsequence"
                    confidence = 0.5
                else:
                    route = "walkthrough_trace_obs_match_history_mismatch"
                    confidence = 0.0
                student_inventory = self._inventory_from_actions(student_history)
                trace_inventory = self._inventory_from_actions(trace_history)
                inventory_match = student_inventory == trace_inventory
                use_action = confidence > 0.0
                # Populate the student-side state signature for state-matched
                # prompts can keep the same teacher payload while only the history
                # matcher (not the structured-state matcher) decides row selection.
                object_match_mode = getattr(self, "recovery_alfworld_object_match_mode", "exact")
                tracked_objects = self._tracked_objects_from_actions(full_actions)
                current_signature = self._execution_state_signature(
                    student_events,
                    state_observation,
                    tracked_objects,
                    object_match_mode,
                )
                result = {
                    "route": route,
                    "action": admissible_by_norm[normalize_action_text(best.next_action)],
                    "continuation": full_actions[best.step_index : best.step_index + self.max_remaining_actions],
                    "confidence": confidence,
                    "use_action": use_action,
                    # Reuse the match_only selection bit: history hits count as matched rows.
                    "state_match": use_action,
                    "match_basis": "history",
                    "state_signature": current_signature,
                    "path_start_index": best.step_index,
                    "trace_history_len": len(trace_history),
                    "inventory_match": inventory_match,
                    "student_inventory_size": len(student_inventory),
                    "trace_inventory_size": len(trace_inventory),
                    "used_last_progress_observation": used_last_progress_observation,
                    "object_match_mode": object_match_mode,
                }
                if self.recovery_correction_only and normalize_action_text(best.next_action) == sampled_action:
                    result.update(
                        {
                            "route": "walkthrough_trace_candidate_matches_student",
                            "action": "",
                            "confidence": 0.0,
                            "use_action": False,
                            "state_match": False,
                            "candidate_matches_student": True,
                        }
                    )
                return result

        start = max(0, min(len(full_actions), consumed))
        search_order = list(range(start, len(full_actions))) + list(range(0, start))
        for action_idx in search_order:
            action = full_actions[action_idx]
            if normalize_action_text(action) in admissible_by_norm:
                route = "walkthrough_action_currently_admissible" if action_idx >= start else "walkthrough_seen_action_currently_admissible"
                return {
                    "route": route,
                    "action": admissible_by_norm[normalize_action_text(action)],
                    "continuation": full_actions[action_idx : action_idx + self.max_remaining_actions],
                    "confidence": 0.0,
                    "use_action": False,
                    "path_start_index": action_idx,
                    "used_last_progress_observation": used_last_progress_observation,
                }

        return {"route": "no_walkthrough_action_currently_admissible", "action": "", "continuation": [], "confidence": 0.0, "use_action": False, "used_last_progress_observation": used_last_progress_observation}

    def _sdl_sample_weight(self, route_name: str, confidence: float) -> float:
        if not self.sdl_confidence_weighting:
            return 1.0
        if route_name in self.sdl_confidence_weights:
            return float(self.sdl_confidence_weights[route_name])
        return float(confidence)

    def _trajectory_success(self, batch: DataProto, idx: int) -> bool:
        """Return the completed trajectory outcome already attached by reward scoring."""
        value = self._value(batch, "episode_rewards", idx, None)
        if value is None:
            value = self._value(batch, "score", idx, 0.0)
        try:
            return float(value or 0.0) > 0.0
        except (TypeError, ValueError):
            return False

    def _apply_success_confirmation_sdl_weight(
        self,
        *,
        batch: DataProto,
        idx: int,
        candidate_action: str,
        sample_weight: float,
    ) -> tuple[float, Dict[str, float]]:
        """Drop/downweight only successful trajectory action confirmations.

        Failure-trajectory confirmations remain active because they can protect
        a locally correct action from coarse trajectory-level credit. Candidate
        selection and the teacher prompt are intentionally left unchanged.
        """
        sampled_action = normalize_action_text(str(self._value(batch, "executed_action", idx, "") or ""))
        candidate_matches = bool(candidate_action) and normalize_action_text(candidate_action) == sampled_action
        trajectory_success = self._trajectory_success(batch, idx)
        success_confirmation = trajectory_success and candidate_matches
        if success_confirmation:
            sample_weight *= self.recovery_success_confirmation_sdl_weight
        return sample_weight, {
            "path_opd/recovery_trajectory_success_ratio": float(trajectory_success),
            "path_opd/recovery_candidate_matches_sampled_ratio": float(candidate_matches),
            "path_opd/recovery_success_confirmation_ratio": float(success_confirmation),
            "path_opd/recovery_success_confirmation_sdl_weight": float(self.recovery_success_confirmation_sdl_weight),
        }

    def _policy_loss_sample_weight(self, sample_metrics: Dict[str, float]) -> float:
        if self.policy_loss_filter in {"repeat_drop", "repeat_after_turn"} and float(sample_metrics.get("path_opd/sdl_repeat_drop_ratio", 0.0) or 0.0) > 0.0:
            return self.policy_loss_repeat_drop_weight
        return 1.0

    def _repeat_drop_active(self, batch: DataProto, idx: int) -> bool:
        if not self.sdl_drop_repeated_action_after_turn:
            return False
        turn_step = int(self._value(batch, "turn_step", idx, 0) or 0)
        if turn_step < self.sdl_repeat_drop_min_turn:
            return False
        sampled_action = normalize_action_text(str(self._value(batch, "executed_action", idx, "") or ""))
        if not sampled_action:
            return False
        history_actions = self._trajectory_prefix_actions(batch, idx)
        return sampled_action in {normalize_action_text(action) for action in history_actions if action}

    def _path_block(
        self,
        batch: DataProto,
        idx: int,
        prompt_text: str = "",
    ) -> tuple[str, Dict[str, float], str]:
        path_candidates = self._path_candidates_from_sample(batch, idx)
        precomputed_recovery: Optional[Dict[str, Any]] = None
        state_candidate_metrics: Dict[str, float] = {}
        if self.path_source_policy == "best_state_aligned_success":
            selected_candidate, precomputed_recovery, state_candidate_metrics = self._select_state_compatible_path_candidate(
                batch=batch,
                idx=idx,
                candidates=path_candidates,
                prompt_text=prompt_text,
            )
        else:
            selected_candidate = self._select_path_candidate(batch, idx, path_candidates)
        full_actions, route = list(selected_candidate.actions), str(selected_candidate.source or "missing")
        nonempty_candidates = [candidate for candidate in path_candidates if candidate.actions]
        buffer_candidates = [candidate for candidate in nonempty_candidates if candidate.source in {"student_autonomous", "assisted_success"}]
        history_actions = self._trajectory_prefix_actions(batch, idx)
        selected_suffix_match = self._longest_suffix_prefix_match(history_actions, full_actions) if full_actions else 0
        selected_prefix_match = self._longest_prefix_match(history_actions, full_actions) if full_actions else 0
        candidate_metrics = {
            "path_opd/path_candidate_count_mean": float(len(nonempty_candidates)),
            "path_opd/path_buffer_candidate_count_mean": float(len(buffer_candidates)),
            "path_opd/path_has_buffer_candidate_ratio": float(bool(buffer_candidates)),
            "path_opd/path_selected_buffer_candidate_ratio": float(route in {"student_autonomous", "assisted_success"}),
            "path_opd/path_selected_suffix_match_mean": float(selected_suffix_match),
            "path_opd/path_selected_prefix_match_mean": float(selected_prefix_match),
            **state_candidate_metrics,
            **self._candidate_filter_metrics.get(idx, {}),
        }
        sample_metadata = self._value(batch, "vmpr_metadata", idx, {}) or {}
        if isinstance(sample_metadata, dict) and sample_metadata.get("webshop_oracle_expected_score") is not None:
            webshop_oracle_score = float(sample_metadata["webshop_oracle_expected_score"])
            candidate_metrics["path_opd/webshop_oracle_expected_score_mean"] = webshop_oracle_score
            candidate_metrics["path_opd/webshop_oracle_full_reward_ratio"] = float(webshop_oracle_score >= 0.999)
        if not full_actions and self.strict_gt_path:
            gamefile = self._value(batch, "gamefile", idx)
            raise ValueError(f"Path-OPD strict_gt_path=True but no GT path was found for sample {idx}, gamefile={gamefile}")
        if not full_actions:
            route_name = route or "missing_gt_path"
            repeat_drop_active = self._repeat_drop_active(batch, idx)
            sample_weight = self.sdl_repeat_drop_weight if repeat_drop_active else 0.0
            return "", {
                f"path_opd/route_{route_name}_ratio": 1.0,
                "path_opd/alignment_missing_path_ratio": 1.0,
                "path_opd/remaining_actions_mean": 0.0,
                "path_opd/consumed_actions_mean": 0.0,
                "path_opd/has_gt_path_ratio": 0.0,
                "path_opd/sdl_sample_weight_mean": float(sample_weight),
                "path_opd/sdl_repeat_drop_ratio": float(repeat_drop_active),
                **candidate_metrics,
            }, ""

        alignment_tier, consumed, executed, remaining = self._alignment_state(batch, idx, full_actions)
        next_action = remaining[0] if remaining else ""
        use_recovery = self.teacher_context in self.RECOVERY_CONTEXTS
        lines = ["[Privileged Path Information]"]
        full_path_text = " -> ".join(full_actions) if full_actions else "(missing)"
        recovery_metrics: Dict[str, float] = {}
        if use_recovery and int(self._value(batch, "turn_step", idx, 0) or 0) == 0 and full_actions and alignment_tier == "full_start":
            alignment_tier, consumed, executed, remaining = "aligned_full_start", 0, [], full_actions[: self.max_remaining_actions]
            next_action = remaining[0] if remaining else ""

        route_name = "other_full_gt_path"
        confidence = 0.0
        sample_weight = self._sdl_sample_weight(route_name, confidence)
        selected_candidate_action = ""
        has_aligned_candidate = alignment_tier in {"aligned_replay", "aligned_full_start", "aligned_history_suffix"}
        if use_recovery and (
            self.recovery_match_mode == "state"
            # State-summary payloads must stay matcher-driven even under history mode,
            # otherwise prefix-aligned turns silently fall back to the old strict
            # oracle renderer and the matcher ablation is no longer single-factor.
            or self.recovery_prompt_style
            in {
                "state_summary_candidate",
                "state_summary_only",
            }
        ):
            # Structured-state matching deliberately routes candidate selection through
            # the recovery matcher instead of accepting action-prefix alignment as
            # sufficient evidence.
            has_aligned_candidate = False
        admissible_actions = extract_admissible_actions(prompt_text)
        admissible_norm = {normalize_action_text(action) for action in admissible_actions}
        # A strict history match is not sufficient evidence by itself: dirty
        # external/student paths can still contain a stale next action.  If the
        # current prompt does not expose a parseable admissible set, fail closed
        # instead of printing an unauditable candidate.
        strict_action_admissible = bool(next_action and admissible_norm and normalize_action_text(next_action) in admissible_norm)
        if has_aligned_candidate and not strict_action_admissible:
            route_name = "strict_next_action_non_admissible"
            confidence = 0.0
            sample_weight = self._sdl_sample_weight(route_name, confidence)
            self._append_full_path(lines, full_path_text)
            self._append_path_guidance(lines)
            if use_recovery:
                recovery_metrics[f"path_opd/recovery_route_{route_name}_ratio"] = 1.0
                recovery_metrics["path_opd/recovery_is_strict_ratio"] = 0.0
                recovery_metrics["path_opd/recovery_strict_action_admissible_ratio"] = 0.0
        elif has_aligned_candidate:
            route_name = "strict_start_next_oracle" if consumed == 0 else "strict_aligned_next_oracle"
            confidence = 1.0
            sample_weight = self._sdl_sample_weight(route_name, confidence)
            selected_candidate_action = next_action
            self._append_full_path(lines, full_path_text)
            lines.append(f"This state matches the successful path after {consumed} action(s).")
            if self.include_executed_prefix:
                lines.append("Matched prefix:")
                lines.append(" -> ".join(executed) if executed else "(none)")
            if next_action:
                lines.append("Candidate next action on the matched path:")
                lines.append(next_action)
            self._append_matched_route_guidance(lines)
            if use_recovery:
                recovery_metrics[f"path_opd/recovery_route_{route_name}_ratio"] = 1.0
                recovery_metrics["path_opd/recovery_is_strict_ratio"] = 1.0
                recovery_metrics["path_opd/recovery_strict_action_admissible_ratio"] = 1.0
        elif use_recovery:
            recovery = precomputed_recovery or self._recovery_guess(
                batch=batch, idx=idx, prompt_text=prompt_text, full_actions=full_actions, consumed=consumed
            )
            route_name = str(recovery.get("route") or "other_full_gt_path")
            guessed_action = str(recovery.get("action") or "")
            guessed_continuation = list(recovery.get("continuation") or [])
            confidence = float(recovery.get("confidence") or 0.0)
            use_guessed_action = bool(recovery.get("use_action", False))
            sample_weight = self._sdl_sample_weight(route_name, confidence)
            selected_candidate_action = guessed_action if use_guessed_action else ""
            goal_progress_prompt = (
                self.state_match_context != "full_path"
                and recovery.get("prompt_mode") == "webshop_goal_progress"
                and bool(recovery.get("state_match", False))
            )
            if self.state_match_context == "full_path":
                # D0@Matched: run the state matcher purely for row selection so
                # that match_only can read recovery_state_match_ratio, but give
                # the teacher the unconditional full-path (D0) context instead of
                # a localized candidate. This isolates the SDL row budget / row
                # selection from localized-context content under a matched budget.
                selected_candidate_action = ""
                self._append_full_path(lines, full_path_text)
                if not self.include_full_path:
                    lines.append(self._full_path_heading())
                    lines.append(full_path_text)
                self._append_path_guidance(lines)
            elif goal_progress_prompt:
                # Full path anchors the teacher; simplified progress/candidate blocks
                # localize the next step (including multi-candidate option sets).
                self._append_full_path(lines, full_path_text)
                if not self.include_full_path:
                    lines.append(self._full_path_heading())
                    lines.append(full_path_text)
                self._append_webshop_goal_progress_guidance(lines, recovery)
            else:
                self._append_full_path(lines, full_path_text)
                match_basis = str(recovery.get("match_basis") or "")
                # State-summary payloads may ride on either matcher. History-mode
                # MatchOnly then isolates coverage/dose while freezing teacher text.
                state_summary_prompt_styles = {"state_summary_candidate", "state_summary_only"}
                use_localized_state_blocks = (
                    use_guessed_action
                    and match_basis == "state"
                    and self.recovery_prompt_style
                    in {
                        "current",
                        "candidate_prefix",
                        "state_summary_candidate",
                        "state_summary_only",
                    }
                )
                use_state_summary_history_blocks = (
                    use_guessed_action
                    and match_basis == "history"
                    and self.recovery_prompt_style in state_summary_prompt_styles
                )
                if use_localized_state_blocks or use_state_summary_history_blocks:
                    path_start_index = int(recovery.get("path_start_index", 0) or 0)
                    state_signature = dict(recovery.get("state_signature") or {})
                    matched_prefix = full_actions[: max(0, min(len(full_actions), path_start_index))]
                    if (
                        self.include_executed_prefix
                        and match_basis == "state"
                        and self.recovery_prompt_style
                        not in state_summary_prompt_styles
                    ):
                        if matched_prefix:
                            if "page_type" in state_signature:
                                lines.append("Your current shopping progress matches the state after these successful-path actions:")
                            else:
                                lines.append("Your current location, inventory, and task progress match the state after these successful-path actions:")
                            lines.append(" -> ".join(matched_prefix))
                        else:
                            if "page_type" in state_signature:
                                lines.append("Your current shopping progress matches the starting state of the successful path.")
                            else:
                                lines.append("Your current location, inventory, and task progress match the starting state of the successful path.")
                            lines.append("Successful-path actions before this state:")
                            lines.append("(none)")
                    if self.recovery_prompt_style in {
                        "current",
                        "state_summary_candidate",
                        "state_summary_only",
                    }:
                        lines.append("Current state summary:")
                        lines.append(self._state_signature_text(state_signature))
                elif (
                    use_guessed_action
                    and match_basis != "state"
                    and self.include_executed_prefix
                    and self.recovery_prompt_style not in state_summary_prompt_styles
                ):
                    path_start_index = int(recovery.get("path_start_index", 0) or 0)
                    matched_prefix = full_actions[: max(0, min(len(full_actions), path_start_index))]
                    lines.append("Matched prefix:")
                    lines.append(" -> ".join(matched_prefix) if matched_prefix else "(none)")
                elif (
                    self.include_executed_prefix
                    and executed
                    and self.recovery_prompt_style not in state_summary_prompt_styles
                    and not (
                        use_guessed_action
                        and match_basis == "state"
                        and self.recovery_prompt_style == "candidate_only"
                    )
                ):
                    lines.append("Matched prefix:")
                    lines.append(" -> ".join(executed))
                if (
                    use_guessed_action
                    and guessed_action
                    and self.recovery_prompt_style != "state_summary_only"
                ):
                    lines.append("Candidate next action for the current state:")
                    lines.append(guessed_action)
                    if (
                        match_basis == "state"
                        and self.recovery_prompt_style == "candidate_only"
                        and self.recovery_guidance_style != "path_on_or_off"
                    ):
                        lines.append(
                            "Use the path information as privileged guidance, but reason from the current observation and admissible actions."
                        )
                    else:
                        self._append_matched_route_guidance(lines)
                else:
                    route_name = route_name if route_name else "other_full_gt_path"
                    if not self.include_full_path:
                        lines.append(self._full_path_heading())
                        lines.append(full_path_text)
                    if (
                        self.recovery_prompt_style == "state_summary_only"
                        and self.recovery_guidance_style != "path_on_or_off"
                    ):
                        lines.append(
                            "Use the path and current state summary as privileged guidance, "
                            "but reason from the current observation and admissible actions."
                        )
                    else:
                        self._append_path_guidance(lines)
            recovery_metrics[f"path_opd/recovery_route_{route_name}_ratio"] = 1.0
            recovery_metrics["path_opd/recovery_inventory_match_ratio"] = float(bool(recovery.get("inventory_match", False)))
            recovery_metrics["path_opd/recovery_state_match_ratio"] = float(bool(recovery.get("state_match", False)))
            recovery_metrics["path_opd/recovery_prompt_style_current"] = float(self.recovery_prompt_style == "current")
            recovery_metrics["path_opd/recovery_prompt_style_candidate_only"] = float(
                self.recovery_prompt_style == "candidate_only"
            )
            recovery_metrics["path_opd/recovery_prompt_style_candidate_prefix"] = float(
                self.recovery_prompt_style == "candidate_prefix"
            )
            recovery_metrics["path_opd/recovery_prompt_style_state_summary_candidate"] = float(
                self.recovery_prompt_style == "state_summary_candidate"
            )
            recovery_metrics["path_opd/recovery_prompt_style_state_summary_only"] = float(
                self.recovery_prompt_style == "state_summary_only"
            )
            recovery_metrics["path_opd/recovery_guidance_style_path_on_or_off"] = float(
                self.recovery_guidance_style == "path_on_or_off"
            )
            recovery_metrics["path_opd/recovery_correction_only_ratio"] = float(self.recovery_correction_only)
            recovery_metrics["path_opd/recovery_state_symbolic_reference_ratio"] = float(bool(recovery.get("symbolic_reference", False)))
            recovery_metrics["path_opd/recovery_alfworld_semantic_instance_mode_ratio"] = float(
                recovery.get("object_match_mode") == "semantic_instance"
            )
            recovery_metrics["path_opd/recovery_semantic_action_grounded_ratio"] = float(
                bool(recovery.get("semantic_action_grounded", False))
            )
            recovery_metrics["path_opd/recovery_grounded_action_count_mean"] = float(
                recovery.get("grounded_action_count", 0) or 0
            )
            state_component_total = max(1, int(recovery.get("state_component_total", 5) or 5))
            recovery_metrics["path_opd/recovery_state_component_match_mean"] = float(recovery.get("state_component_match_count", 0) or 0) / float(state_component_total)
            state_component_match = dict(recovery.get("state_component_match") or {})
            for component in ("location", "inventory", "object_locations", "object_properties", "activated_tools"):
                recovery_metrics[f"path_opd/recovery_state_{component}_match_ratio"] = float(bool(state_component_match.get(component, False)))
            for component in ("page_type", "asin", "options"):
                recovery_metrics[f"path_opd/recovery_webshop_{component}_match_ratio"] = float(bool(state_component_match.get(component, False)))
            guidance_kind = str(recovery.get("guidance_kind") or "none")
            recovery_metrics["path_opd/recovery_webshop_goal_progress_ratio"] = float(
                recovery.get("prompt_mode") == "webshop_goal_progress" and bool(recovery.get("state_match", False))
            )
            recovery_metrics["path_opd/recovery_webshop_unique_candidate_ratio"] = float(
                guidance_kind == "unique_action" and bool(use_guessed_action and guessed_action)
            )
            recovery_metrics["path_opd/recovery_webshop_set_guidance_ratio"] = float(
                guidance_kind in {"option_set", "search_query_set"}
            )
            recovery_metrics["path_opd/recovery_webshop_selected_options_mean"] = float(
                len(dict(recovery.get("selected_options") or {}))
            )
            recovery_metrics["path_opd/recovery_webshop_remaining_options_mean"] = float(
                len(dict(recovery.get("remaining_options") or {}))
            )
            recovery_metrics["path_opd/recovery_webshop_candidate_set_size_mean"] = float(
                len(list(recovery.get("candidate_actions") or []))
            )
            recovery_metrics["path_opd/recovery_trace_history_len_mean"] = float(recovery.get("trace_history_len", 0) or 0)
            recovery_metrics["path_opd/recovery_used_last_progress_observation_ratio"] = float(bool(recovery.get("used_last_progress_observation", False)))
        elif alignment_tier in {"deviated_replay", "deviated_full_start"}:
            self._append_full_path(lines, full_path_text)
            self._append_path_guidance(lines)
        elif alignment_tier == "legacy":
            if self.include_executed_prefix:
                lines.append("Matched prefix:")
                lines.append(" -> ".join(executed) if executed else "(none)")
            self._append_full_path(lines, full_path_text)
            self._append_path_guidance(lines)
        else:
            self._append_full_path(lines, full_path_text)
            if not self.include_full_path:
                lines.append(self._full_path_heading())
                lines.append(" -> ".join(remaining) if remaining else "(missing)")
            self._append_path_guidance(lines)
        success_confirmation_metrics: Dict[str, float] = {}
        if use_recovery:
            sample_weight, success_confirmation_metrics = self._apply_success_confirmation_sdl_weight(
                batch=batch,
                idx=idx,
                candidate_action=selected_candidate_action,
                sample_weight=sample_weight,
            )
        repeat_drop_active = self._repeat_drop_active(batch, idx)
        if repeat_drop_active:
            sample_weight = self.sdl_repeat_drop_weight
        lines.append("[/Privileged Path Information]")
        return "\n".join(lines), {
            f"path_opd/route_{route}_ratio": 1.0,
            f"path_opd/alignment_{alignment_tier}_ratio": 1.0,
            "path_opd/remaining_actions_mean": float(len(remaining)),
            "path_opd/consumed_actions_mean": float(consumed),
            "path_opd/has_gt_path_ratio": float(bool(full_actions)),
            "path_opd/sdl_sample_weight_mean": float(sample_weight),
            "path_opd/sdl_repeat_drop_ratio": float(repeat_drop_active),
            **candidate_metrics,
            **recovery_metrics,
            **success_confirmation_metrics,
        }, selected_candidate_action

    def build_context(self, *, batch: DataProto, idx: int, prompt_text: str) -> str:
        gamefile = self._value(batch, "gamefile", idx)
        data_source = self._value(batch, "data_source", idx)
        blocks = []
        metrics = {}
        candidate_action = ""
        if self.teacher_context in self.STATE_CONTEXTS:
            if ("[History Summary]" in prompt_text and "[/History Summary]" in prompt_text) or ("[State]" in prompt_text and "[/State]" in prompt_text):
                metrics["path_opd/reused_student_state_ratio"] = 1.0
            else:
                block, state_metrics = self._state_history_block(batch, idx, prompt_text=prompt_text)
                blocks.append(block)
                metrics.update(state_metrics)
        if self.teacher_context in self.PATH_CONTEXTS:
            block, path_metrics, candidate_action = self._path_block(batch, idx, prompt_text=prompt_text)
            if block:
                blocks.append(block)
            metrics.update(path_metrics)
        if self.teacher_context in self.SKILL_CONTEXTS:
            skill_text = self._skill_text(prompt_text=prompt_text, gamefile=gamefile, data_source=data_source)
            if skill_text:
                blocks.append(f"[Privileged Skill Information]\n{skill_text}")
                metrics["path_opd/has_skill_ratio"] = 1.0
                skill_task = "unknown"
                if self.skill_provider is not None:
                    if gamefile:
                        skill_task = self.skill_provider.infer_task_type_from_gamefile(str(gamefile)) or "general_only"
                    else:
                        skill_task = self.skill_provider.infer_task_type_from_prompt(prompt_text) or "general_only"
                metrics[f"path_opd/skill_task_{skill_task}_ratio"] = 1.0
                metrics["path_opd/skill_task_specific_ratio"] = float(skill_task not in {"unknown", "general_only"})
            else:
                metrics["path_opd/has_skill_ratio"] = 0.0
                metrics["path_opd/skill_task_specific_ratio"] = 0.0
        if not blocks:
            metrics["path_opd/no_context_ratio"] = 1.0
        context_text = "\n\n".join(blocks)
        self._sample_metrics.append(metrics)
        self._sample_contexts.append(context_text)
        self._sample_candidate_actions.append(candidate_action)
        return context_text

    def start_batch_metrics(self) -> None:
        self._sample_metrics: List[Dict[str, float]] = []
        self._sample_contexts: List[str] = []
        self._sample_candidate_actions: List[str] = []
        self._candidate_filter_metrics = {}
        self._reset_batch_caches()

    def finish_batch_metrics(self) -> Dict[str, float]:
        if not self._sample_metrics:
            self.last_metrics = {}
            return {}
        totals: Dict[str, float] = {}
        for metrics in self._sample_metrics:
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
        count = float(len(self._sample_metrics))
        self.last_metrics = {key: value / count for key, value in totals.items()}
        self.last_metrics["path_opd/samples"] = count
        self.last_metrics["path_opd/trajectory_index_size"] = float(len(self.trajectory_index.by_key))
        self.last_metrics["path_opd/path_buffer_index_size"] = float(self.path_buffer_index.candidate_count)
        self.last_metrics["path_opd/path_buffer_reload_count"] = float(self.path_buffer_index.reload_count)
        self.last_metrics["path_opd/walkthrough_trace_cache_size"] = float(self.trace_cache.cache_size)
        self.last_metrics["path_opd/walkthrough_trace_build_count"] = float(self.trace_cache.build_count)
        self.last_metrics["path_opd/walkthrough_trace_error_count"] = float(self.trace_cache.error_count)
        return self.last_metrics


def insert_privileged_context(prompt_text: str, context_text: str) -> str:
    if not context_text:
        return prompt_text
    if context_text.lstrip().startswith("[History Summary]"):
        history_text = context_text
        remaining_context = ""
        end_marker = "[/History Summary]"
        end_idx = context_text.find(end_marker)
        if end_idx >= 0:
            split_idx = end_idx + len(end_marker)
            history_text = context_text[:split_idx].strip()
            remaining_context = context_text[split_idx:].strip()
        prior_match = re.search(r"(Prior to this step, you have already taken \d+ step\(s\)\.)", prompt_text)
        if prior_match:
            insert_at = prior_match.end()
            prompt_text = f"{prompt_text[:insert_at]}\n\n{history_text}\n\n{prompt_text[insert_at:].lstrip()}"
        else:
            marker = "\nYou are now at step "
            if marker in prompt_text:
                prompt_text = prompt_text.replace(marker, f"\n\n{history_text}\n{marker}", 1)
            else:
                fallback_marker = "\nYour admissible actions of the current situation are:"
                if fallback_marker in prompt_text:
                    prompt_text = prompt_text.replace(fallback_marker, f"\n\n{history_text}\n{fallback_marker}", 1)
                else:
                    prompt_text = f"{history_text}\n\n{prompt_text}"
        if remaining_context:
            context_block = f"\n\n{remaining_context}\n"
            marker = "\n\nNow it's your turn to take an action."
            if marker in prompt_text:
                prompt_text = prompt_text.replace(marker, context_block + marker, 1)
            else:
                prompt_text = f"{prompt_text}\n\n{remaining_context}"
        return make_history_reasoning_instruction(prompt_text)

    context_block = f"\n\n{context_text}\n"
    marker = "\n\nNow it's your turn to take an action."
    if marker in prompt_text:
        return prompt_text.replace(marker, context_block + marker, 1)

    user_start = "<|im_start|>user\n"
    if user_start in prompt_text:
        return prompt_text.replace(user_start, user_start + f"{context_text}\n\n", 1)
    return f"{context_text}\n\n{prompt_text}"


def make_history_reasoning_instruction(prompt_text: str) -> str:
    no_think_old = (
        "Now it's your turn to take an action.\n"
        "You should first reason step-by-step about the current situation.\n"
        "Once you've finished your reasoning, you should choose exactly one admissible action for current step and present it within <action> </action> tags."
    )
    no_think_new = (
        "Now it's your turn to take an action.\n"
        "You should first reason step-by-step about the current situation using the current observation, the History Summary, and the admissible actions.\n"
        "Use the History Summary to track what has already been tried and avoid unproductive repeats.\n"
        "Once you've finished your reasoning, you should choose exactly one admissible action for current step and present it within <action> </action> tags."
    )
    if no_think_old in prompt_text:
        return prompt_text.replace(no_think_old, no_think_new, 1)

    think_old = (
        "Now it's your turn to take an action.\n"
        "You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. \n"
        "Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."
    )
    think_new = (
        "Now it's your turn to take an action.\n"
        "You should first reason step-by-step about the current situation using the current observation, the History Summary, and the admissible actions. This reasoning process MUST be enclosed within <think> </think> tags. \n"
        "Use the History Summary to track what has already been tried and avoid unproductive repeats.\n"
        "Once you've finished your reasoning, you should choose exactly one admissible action for current step and present it within <action> </action> tags."
    )
    if think_old in prompt_text:
        return prompt_text.replace(think_old, think_new, 1)

    action_only_old = (
        "Now it's your turn to take an action.\n"
        "Choose exactly one admissible action for current step and present it within <action> </action> tags."
    )
    action_only_new = (
        "Now it's your turn to take an action.\n"
        "Briefly use the current observation, the History Summary, and the admissible actions to avoid unproductive repeats.\n"
        "Choose exactly one admissible action for current step and present it within <action> </action> tags."
    )
    if action_only_old in prompt_text:
        return prompt_text.replace(action_only_old, action_only_new, 1)
    return prompt_text


def build_state_sdl_selection_mask(
    sample_metrics: list[dict[str, float]],
    *,
    mode: str,
    random_seed: int = 0,
    global_step: int = 0,
) -> tuple[list[float], dict[str, float]]:
    """Select SDL rows for state-compatibility budget ablations.

    Selection happens once on the complete teacher batch, before actor
    micro-batching. ``random_equal`` and ``unsupported_equal`` therefore have
    exactly the same per-step row budget as ``match_only`` and sample without
    replacement. ``random_equal`` samples from the complete batch, whereas
    ``unsupported_equal`` samples only from the matcher's complement. The
    returned mask only changes SDL sample weights; policy/GRPO weights stay
    independent.
    """
    normalized_mode = str(mode or "all").lower()
    supported_modes = {
        "all",
        "match_only",
        "random_equal",
        "mismatch_only",
        "unsupported_equal",
    }
    if normalized_mode not in supported_modes:
        raise ValueError(
            f"Unsupported state SDL selection mode={normalized_mode!r}. "
            "Expected all, match_only, random_equal, mismatch_only, or unsupported_equal."
        )

    sample_count = len(sample_metrics)
    match_mask = [
        float(float(metrics.get("path_opd/recovery_state_match_ratio", 0.0) or 0.0) > 0.5)
        for metrics in sample_metrics
    ]
    match_count = int(sum(match_mask))
    unsupported_indices = [
        index for index, matched in enumerate(match_mask) if not matched
    ]
    unsupported_count = len(unsupported_indices)

    if normalized_mode == "all":
        selection_mask = [1.0] * sample_count
    elif normalized_mode == "match_only":
        selection_mask = match_mask
    elif normalized_mode == "mismatch_only":
        selection_mask = [1.0 - matched for matched in match_mask]
    else:
        selection_mask = [0.0] * sample_count
        if match_count:
            candidate_indices = (
                unsupported_indices
                if normalized_mode == "unsupported_equal"
                else list(range(sample_count))
            )
            if match_count > len(candidate_indices):
                raise ValueError(
                    "unsupported_equal cannot satisfy the exact per-step "
                    "MatchOnly-sized budget without replacement: "
                    f"match_count={match_count}, unsupported_count={unsupported_count}, "
                    f"sample_count={sample_count}"
                )
            generator = torch.Generator(device="cpu")
            step_seed = (int(random_seed) + 1_000_003 * int(global_step)) % (2**63 - 1)
            generator.manual_seed(step_seed)
            selected_offsets = torch.randperm(
                len(candidate_indices), generator=generator
            )[:match_count].tolist()
            for offset in selected_offsets:
                selection_mask[candidate_indices[offset]] = 1.0

    selected_count = int(sum(selection_mask))
    overlap_count = int(sum(bool(selected and matched) for selected, matched in zip(selection_mask, match_mask)))
    unsupported_overlap_count = selected_count - overlap_count
    sample_denominator = float(max(1, sample_count))
    return selection_mask, {
        "path_opd/state_sdl_selection_all": float(normalized_mode == "all"),
        "path_opd/state_sdl_selection_match_only": float(normalized_mode == "match_only"),
        "path_opd/state_sdl_selection_random_equal": float(normalized_mode == "random_equal"),
        "path_opd/state_sdl_selection_mismatch_only": float(normalized_mode == "mismatch_only"),
        "path_opd/state_sdl_selection_unsupported_equal": float(
            normalized_mode == "unsupported_equal"
        ),
        "path_opd/state_sdl_match_count": float(match_count),
        "path_opd/state_sdl_match_ratio": float(match_count / sample_denominator),
        "path_opd/state_sdl_unsupported_count": float(unsupported_count),
        "path_opd/state_sdl_unsupported_ratio": float(
            unsupported_count / sample_denominator
        ),
        "path_opd/state_sdl_selected_count": float(selected_count),
        "path_opd/state_sdl_selected_ratio": float(selected_count / sample_denominator),
        "path_opd/state_sdl_selected_minus_match_count": float(selected_count - match_count),
        "path_opd/state_sdl_selected_match_overlap_count": float(overlap_count),
        "path_opd/state_sdl_selected_match_overlap_ratio": float(overlap_count / max(1, match_count)),
        "path_opd/state_sdl_selected_unsupported_overlap_count": float(
            unsupported_overlap_count
        ),
        "path_opd/state_sdl_selected_unsupported_overlap_ratio": float(
            unsupported_overlap_count / max(1, unsupported_count)
        ),
    }


def select_public_path_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Keep only method-facing routing and selection health metrics.

    Per-sample prompts, confidence heuristics, teacher-gap probes, timing, and
    replay/debug counters are research diagnostics rather than reproducibility
    outputs.  The retained fields verify the state matcher and the SDL router.
    """

    exact_keys = {
        "path_opd/route_trajectory_index_ratio",
        "path_opd/has_gt_path_ratio",
        "path_opd/recovery_prompt_style_state_summary_candidate",
        "path_opd/recovery_guidance_style_path_on_or_off",
        "path_opd/recovery_webshop_goal_progress_ratio",
        "path_opd/recovery_webshop_set_guidance_ratio",
    }
    prefixes = (
        "path_opd/state_sdl_",
        "path_opd/recovery_route_",
        "path_opd/recovery_state_",
        "path_opd/recovery_webshop_page_type_match_ratio",
        "path_opd/recovery_webshop_asin_match_ratio",
        "path_opd/recovery_webshop_options_match_ratio",
    )
    return {
        key: value
        for key, value in metrics.items()
        if key in exact_keys or key.startswith(prefixes)
    }


def build_path_privileged_teacher_batch(
    *,
    batch: DataProto,
    context_provider: PathPrivilegedContextProvider,
    tokenizer,
    max_prompt_length: int,
    global_step: int = 0,
) -> DataProto:
    bs = batch.batch["input_ids"].size(0)
    response_length = batch.batch["responses"].size(1)
    teacher_input_ids_list = []
    teacher_attention_mask_list = []
    teacher_position_ids_list = []
    context_provider.start_batch_metrics()

    for i in range(bs):
        original_input_ids = batch.batch["input_ids"][i]
        original_attention_mask = batch.batch["attention_mask"][i]
        prompt_length = original_input_ids.size(0) - response_length
        prompt_ids = original_input_ids[:prompt_length]
        prompt_mask = original_attention_mask[:prompt_length]
        valid_start = prompt_mask.nonzero(as_tuple=True)[0]
        valid_start = valid_start[0].item() if len(valid_start) > 0 else 0
        prompt_text = tokenizer.decode(prompt_ids[valid_start:], skip_special_tokens=False)

        context_text = context_provider.build_context(batch=batch, idx=i, prompt_text=prompt_text)
        teacher_prompt_text = insert_privileged_context(prompt_text, context_text)

        teacher_prompt_ids = tokenizer.encode(teacher_prompt_text, add_special_tokens=False)
        if len(teacher_prompt_ids) > max_prompt_length:
            teacher_prompt_ids = teacher_prompt_ids[-max_prompt_length:]

        teacher_prompt_ids = torch.tensor(teacher_prompt_ids, dtype=torch.long)
        actual_prompt_len = len(teacher_prompt_ids)
        pad_length = max_prompt_length - actual_prompt_len
        if pad_length > 0:
            pad_ids = torch.full((pad_length,), tokenizer.pad_token_id, dtype=torch.long)
            teacher_prompt_ids = torch.cat([pad_ids, teacher_prompt_ids])
            t_prompt_mask = torch.cat([torch.zeros(pad_length, dtype=torch.long), torch.ones(actual_prompt_len, dtype=torch.long)])
        else:
            t_prompt_mask = torch.ones(actual_prompt_len, dtype=torch.long)

        response_ids = batch.batch["responses"][i]
        response_mask = original_attention_mask[-response_length:]
        teacher_full_ids = torch.cat([teacher_prompt_ids, response_ids])
        teacher_full_mask = torch.cat([t_prompt_mask, response_mask])
        teacher_position_ids = compute_position_id_with_mask(teacher_full_mask.unsqueeze(0))[0]

        teacher_input_ids_list.append(teacher_full_ids)
        teacher_attention_mask_list.append(teacher_full_mask)
        teacher_position_ids_list.append(teacher_position_ids)

    metrics = select_public_path_metrics(context_provider.finish_batch_metrics())
    sdl_sample_weights = [float(sample_metrics.get("path_opd/sdl_sample_weight_mean", 1.0)) for sample_metrics in context_provider._sample_metrics]
    state_selection_mask, state_selection_metrics = build_state_sdl_selection_mask(
        context_provider._sample_metrics,
        mode=context_provider.state_sdl_selection,
        random_seed=context_provider.state_sdl_random_seed,
        global_step=global_step,
    )
    if len(state_selection_mask) != len(sdl_sample_weights):
        raise RuntimeError("State SDL selection mask and teacher batch size disagree.")
    sdl_sample_weights = [weight * selected for weight, selected in zip(sdl_sample_weights, state_selection_mask)]
    metrics.update(select_public_path_metrics(state_selection_metrics))
    sdl_sample_weight = torch.tensor(sdl_sample_weights, dtype=torch.float32)
    policy_loss_sample_weights = [float(context_provider._policy_loss_sample_weight(sample_metrics)) for sample_metrics in context_provider._sample_metrics]
    policy_loss_sample_weight = torch.tensor(policy_loss_sample_weights, dtype=torch.float32)
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.stack(teacher_input_ids_list),
            "attention_mask": torch.stack(teacher_attention_mask_list),
            "position_ids": torch.stack(teacher_position_ids_list),
            "responses": batch.batch["responses"],
        },
        meta_info={"teacher_skill_metrics": metrics, "sdl_sample_weight": sdl_sample_weight, "policy_loss_sample_weight": policy_loss_sample_weight},
    )


def build_path_candidate_ce_batch(
    *,
    batch: DataProto,
    context_provider: PathPrivilegedContextProvider,
    tokenizer,
    max_response_length: int,
    global_step: int = 0,
) -> DataProto:
    """Build ordinary-prompt, action-only targets from the path matcher.

    This is the direct-supervision control for Path-OPD. It deliberately runs
    the exact same context provider as the privileged teacher, but discards the
    rendered privileged context. The student instead sees its original prompt
    and is trained to emit only ``<action>{candidate}</action>`` on rows selected
    by ``state_sdl_selection``.
    """
    if max_response_length <= 0:
        raise ValueError(f"candidate_ce_max_response_length must be positive, got {max_response_length}")
    bs = batch.batch["input_ids"].size(0)
    sampled_response_length = batch.batch["responses"].size(1)
    prompt_length = batch.batch["input_ids"].size(1) - sampled_response_length
    if prompt_length <= 0:
        raise ValueError(
            "Candidate CE requires input_ids to contain a non-empty prompt before the sampled response."
        )

    candidate_input_ids_list = []
    candidate_attention_mask_list = []
    candidate_position_ids_list = []
    candidate_responses_list = []
    candidate_response_mask_list = []
    target_token_counts = []
    context_provider.start_batch_metrics()

    for i in range(bs):
        original_input_ids = batch.batch["input_ids"][i]
        original_attention_mask = batch.batch["attention_mask"][i]
        prompt_ids = original_input_ids[:prompt_length]
        prompt_mask = original_attention_mask[:prompt_length]
        valid_start = prompt_mask.nonzero(as_tuple=True)[0]
        valid_start = valid_start[0].item() if len(valid_start) > 0 else 0
        prompt_text = tokenizer.decode(prompt_ids[valid_start:], skip_special_tokens=False)

        # build_context is the single source of truth for matcher routing and
        # the selected candidate. Its rendered privileged text is intentionally
        # not inserted into the candidate-CE prompt.
        context_provider.build_context(batch=batch, idx=i, prompt_text=prompt_text)
        candidate_action = str(context_provider._sample_candidate_actions[-1] or "").strip()
        target_text = f"<action>{candidate_action}</action>" if candidate_action else ""
        target_ids = tokenizer.encode(target_text, add_special_tokens=False) if target_text else []
        if len(target_ids) > max_response_length:
            raise ValueError(
                "Candidate CE target exceeds algorithm.path_opd.candidate_ce_max_response_length: "
                f"sample={i}, target_tokens={len(target_ids)}, max={max_response_length}, "
                f"candidate={candidate_action!r}"
            )

        target_tensor = torch.full(
            (max_response_length,),
            tokenizer.pad_token_id,
            dtype=original_input_ids.dtype,
            device=original_input_ids.device,
        )
        target_mask = torch.zeros(
            max_response_length,
            dtype=original_attention_mask.dtype,
            device=original_attention_mask.device,
        )
        if target_ids:
            target_tensor[: len(target_ids)] = torch.tensor(
                target_ids,
                dtype=original_input_ids.dtype,
                device=original_input_ids.device,
            )
            target_mask[: len(target_ids)] = 1

        candidate_full_ids = torch.cat([prompt_ids, target_tensor])
        candidate_full_mask = torch.cat([prompt_mask, target_mask])
        candidate_position_ids = compute_position_id_with_mask(candidate_full_mask.unsqueeze(0))[0]

        candidate_input_ids_list.append(candidate_full_ids)
        candidate_attention_mask_list.append(candidate_full_mask)
        candidate_position_ids_list.append(candidate_position_ids)
        candidate_responses_list.append(target_tensor)
        candidate_response_mask_list.append(target_mask)
        target_token_counts.append(float(len(target_ids)))

    metrics = select_public_path_metrics(context_provider.finish_batch_metrics())

    candidate_sample_weights = [
        float(sample_metrics.get("path_opd/sdl_sample_weight_mean", 1.0))
        for sample_metrics in context_provider._sample_metrics
    ]
    state_selection_mask, state_selection_metrics = build_state_sdl_selection_mask(
        context_provider._sample_metrics,
        mode=context_provider.state_sdl_selection,
        random_seed=context_provider.state_sdl_random_seed,
        global_step=global_step,
    )
    if len(state_selection_mask) != len(candidate_sample_weights):
        raise RuntimeError("State SDL selection mask and candidate CE batch size disagree.")
    candidate_available_mask = [float(bool(action)) for action in context_provider._sample_candidate_actions]
    candidate_sample_weights = [
        weight * selected * available
        for weight, selected, available in zip(
            candidate_sample_weights,
            state_selection_mask,
            candidate_available_mask,
        )
    ]
    metrics.update(select_public_path_metrics(state_selection_metrics))

    policy_loss_sample_weights = [
        float(context_provider._policy_loss_sample_weight(sample_metrics))
        for sample_metrics in context_provider._sample_metrics
    ]
    candidate_response_mask = torch.stack(candidate_response_mask_list)
    normalization_token_count = batch.batch["response_mask"].float().sum(dim=-1)
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.stack(candidate_input_ids_list),
            "attention_mask": torch.stack(candidate_attention_mask_list),
            "position_ids": torch.stack(candidate_position_ids_list),
            "responses": torch.stack(candidate_responses_list),
            "candidate_response_mask": candidate_response_mask,
            "candidate_sample_weight": torch.tensor(candidate_sample_weights, dtype=torch.float32),
            "normalization_token_count": normalization_token_count,
        },
        meta_info={
            "candidate_ce_metrics": metrics,
            "policy_loss_sample_weight": torch.tensor(policy_loss_sample_weights, dtype=torch.float32),
        },
    )
