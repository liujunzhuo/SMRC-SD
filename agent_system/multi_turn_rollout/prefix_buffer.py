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

import json
import os
import random
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


TEACHER_SOURCE = "teacher_seed"
STUDENT_SOURCE = "student_autonomous"
ASSISTED_SOURCE = "assisted_success"
FULL_START_SOURCE = "full_start"


def _relocate_alfworld_path(path: Optional[str]) -> Optional[str]:
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
    relocated = os.path.join(os.path.expanduser(alfworld_data), marker, suffix)
    return relocated if os.path.exists(relocated) else expanded


def _relocate_prefix_payload_paths(payload: Dict[str, Any]) -> Dict[str, Any]:
    reset_kwargs = dict(payload.get("reset_kwargs") or {})
    if reset_kwargs.get("gamefile"):
        reset_kwargs["gamefile"] = _relocate_alfworld_path(reset_kwargs["gamefile"])
        payload["reset_kwargs"] = reset_kwargs
    if payload.get("task_key"):
        payload["task_key"] = _relocate_alfworld_path(payload["task_key"])
    if payload.get("gamefile"):
        payload["gamefile"] = _relocate_alfworld_path(payload["gamefile"])
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("traj_data"):
        metadata["traj_data"] = _relocate_alfworld_path(metadata["traj_data"])
        payload["metadata"] = metadata
    return payload


@dataclass
class PrefixEntry:
    prefix_id: str
    source: str
    env_name: str
    task_key: str
    reset_kwargs: Dict[str, Any]
    prefix_actions: List[str]
    boundary_type: str
    prefix_len: int
    insert_step: int
    last_used_step: int = 0
    use_count: int = 0
    verified: bool = False
    replay_fail_count: int = 0
    success_ema: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrefixEntry":
        payload = _relocate_prefix_payload_paths(dict(data))
        payload.setdefault("prefix_id", str(uuid.uuid4()))
        payload.setdefault("source", TEACHER_SOURCE)
        payload.setdefault("env_name", "")
        payload.setdefault("task_key", "")
        payload.setdefault("reset_kwargs", {})
        payload.setdefault("prefix_actions", [])
        payload.setdefault("boundary_type", "unknown")
        payload["prefix_len"] = int(payload.get("prefix_len", len(payload["prefix_actions"])))
        payload.setdefault("insert_step", 0)
        payload.setdefault("last_used_step", 0)
        payload.setdefault("use_count", 0)
        payload.setdefault("verified", False)
        payload.setdefault("replay_fail_count", 0)
        payload.setdefault("success_ema", 0.0)
        payload.setdefault("metadata", {})
        return cls(**payload)

    def to_reset_kwargs(self) -> Dict[str, Any]:
        reset_kwargs = dict(self.reset_kwargs or {})
        reset_kwargs.update(
            {
                "vmpr_prefix_id": self.prefix_id,
                "vmpr_source": self.source,
                "prefix_actions": list(self.prefix_actions),
                "vmpr_metadata": dict(self.metadata or {}),
            }
        )
        return reset_kwargs


class PrefixBuffer:
    def __init__(
        self,
        *,
        env_name: str,
        seed_prefix_path: Optional[str] = None,
        buffer_path: Optional[str] = None,
        max_buffer_size: int = 4096,
        step_boundaries: Optional[Iterable[int]] = None,
        teacher_source_weight: float = 1.0,
        student_source_weight: float = 1.0,
        assisted_source_weight: float = 0.2,
        max_teacher_buffer_size: Optional[int] = None,
        max_student_buffer_size: Optional[int] = None,
        max_assisted_buffer_size: Optional[int] = None,
        preserve_teacher_seed: bool = False,
        source_sampling_strategy: str = "entry_weighted",
        max_prefix_use_count: int = 0,
        max_student_prefix_use_count: int = 0,
        drop_prefix_after_sample: bool = False,
        success_ema_sampling_weight: float = 0.0,
        freshness_decay: float = 0.001,
        student_insertion_freshness_decay: float = 0.0,
        student_prefix_boundary_sample: str = "all",
        max_student_prefix_length_ratio: float = 0.0,
        seed: int = 0,
    ):
        self.env_name = env_name
        self.buffer_path = os.path.expanduser(buffer_path) if buffer_path else None
        self.max_buffer_size = int(max_buffer_size)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.step_boundaries = sorted({int(x) for x in (step_boundaries or []) if int(x) > 0})
        self.source_weights = {
            TEACHER_SOURCE: float(teacher_source_weight),
            STUDENT_SOURCE: float(student_source_weight),
            ASSISTED_SOURCE: float(assisted_source_weight),
        }
        self.source_limits = {
            TEACHER_SOURCE: self._optional_int(max_teacher_buffer_size),
            STUDENT_SOURCE: self._optional_int(max_student_buffer_size),
            ASSISTED_SOURCE: self._optional_int(max_assisted_buffer_size),
        }
        self.preserve_teacher_seed = self._optional_bool(preserve_teacher_seed)
        self.source_sampling_strategy = str(source_sampling_strategy or "entry_weighted")
        if self.source_sampling_strategy not in {"entry_weighted", "source_first"}:
            raise ValueError(f"Unsupported prefix source_sampling_strategy={self.source_sampling_strategy}. Expected entry_weighted or source_first.")
        self.max_prefix_use_count = int(max_prefix_use_count or 0)
        self.max_student_prefix_use_count = int(max_student_prefix_use_count or 0)
        self.drop_prefix_after_sample = self._optional_bool(drop_prefix_after_sample)
        self.success_ema_sampling_weight = float(success_ema_sampling_weight or 0.0)
        self.freshness_decay = float(freshness_decay)
        self.student_insertion_freshness_decay = float(student_insertion_freshness_decay or 0.0)
        self.student_prefix_boundary_sample = str(student_prefix_boundary_sample or "all")
        if self.student_prefix_boundary_sample not in {"all", "random_one"}:
            raise ValueError(f"Unsupported student_prefix_boundary_sample={self.student_prefix_boundary_sample}. Expected all or random_one.")
        self.max_student_prefix_length_ratio = float(max_student_prefix_length_ratio or 0.0)
        if self.max_student_prefix_length_ratio < 0:
            raise ValueError("max_student_prefix_length_ratio must be non-negative.")
        self.entries: Dict[str, PrefixEntry] = {}
        self._seen_prefix_signatures = set()
        self.last_add_duplicate_count = 0
        self.last_sample_stats: Optional[Dict[str, Any]] = None
        self._load_jsonl(seed_prefix_path, default_source=TEACHER_SOURCE)
        self._load_jsonl(self.buffer_path, default_source=None)

    @staticmethod
    def _optional_int(value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, str):
            if value.lower() in {"none", "null", ""}:
                return None
            value = int(value)
        return int(value)

    @staticmethod
    def _optional_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _load_jsonl(self, path: Optional[str], default_source: Optional[str]) -> None:
        if not path:
            return
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if default_source is not None:
                    data["source"] = default_source
                entry = PrefixEntry.from_dict(data)
                signature = self._entry_signature(entry)
                if signature in self._seen_prefix_signatures:
                    continue
                self.entries[entry.prefix_id] = entry
                self._seen_prefix_signatures.add(signature)
        self._trim()

    def save(self) -> None:
        if not self.buffer_path:
            return
        os.makedirs(os.path.dirname(self.buffer_path) or ".", exist_ok=True)
        with open(self.buffer_path, "w", encoding="utf-8") as f:
            for entry in self.entries.values():
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self.entries)

    def source_counts(self) -> Dict[str, int]:
        counts = {TEACHER_SOURCE: 0, STUDENT_SOURCE: 0, ASSISTED_SOURCE: 0}
        for entry in self.entries.values():
            counts[entry.source] = counts.get(entry.source, 0) + 1
        return counts

    def prefix_len_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for entry in self.entries.values():
            counts[entry.prefix_len] = counts.get(entry.prefix_len, 0) + 1
        return counts

    def age_stats(self, *, global_step: int) -> Dict[str, float]:
        if not self.entries:
            return {}
        ages = sorted(float(max(0, global_step - entry.insert_step)) for entry in self.entries.values())
        count = len(ages)
        return {
            "mean": sum(ages) / count,
            "p50": ages[min(count - 1, int(0.50 * (count - 1)))],
            "p90": ages[min(count - 1, int(0.90 * (count - 1)))],
        }

    def _entry_sampling_weight(self, entry: PrefixEntry, *, global_step: int, include_source_weight: bool) -> float:
        source_weight = self.source_weights.get(entry.source, 0.0) if include_source_weight else 1.0
        if source_weight <= 0:
            return 0.0
        if entry.source == STUDENT_SOURCE and self.student_insertion_freshness_decay > 0:
            age = max(0, global_step - entry.insert_step)
            freshness = 1.0 / (1.0 + age * self.student_insertion_freshness_decay)
        else:
            freshness = 1.0 / (1.0 + max(0, global_step - entry.last_used_step) * self.freshness_decay)
        fail_penalty = 1.0 / (1.0 + entry.replay_fail_count)
        success_weight = max(0.0, 1.0 + self.success_ema_sampling_weight * float(entry.success_ema))
        return source_weight * freshness * fail_penalty * success_weight

    def _remove_after_sample_if_needed(self, entry: PrefixEntry) -> None:
        should_drop = self.drop_prefix_after_sample
        if self.max_prefix_use_count > 0 and entry.use_count >= self.max_prefix_use_count:
            should_drop = True
        if entry.source == STUDENT_SOURCE and self.max_student_prefix_use_count > 0 and entry.use_count >= self.max_student_prefix_use_count:
            should_drop = True
        if should_drop:
            self.entries.pop(entry.prefix_id, None)

    def sample(
        self,
        *,
        global_step: int,
        min_prefix_len: Optional[int] = None,
        max_prefix_len: Optional[int] = None,
        prefer_longest_below_max: bool = False,
    ) -> Optional[PrefixEntry]:
        self.last_sample_stats = None
        candidates_by_source: Dict[str, List[PrefixEntry]] = {}
        weights_by_source: Dict[str, List[float]] = {}
        for entry in self.entries.values():
            if min_prefix_len is not None and entry.prefix_len < min_prefix_len:
                continue
            if max_prefix_len is not None and entry.prefix_len > max_prefix_len:
                continue
            weight = self._entry_sampling_weight(entry, global_step=global_step, include_source_weight=self.source_sampling_strategy == "entry_weighted")
            if weight <= 0:
                continue
            candidates_by_source.setdefault(entry.source, []).append(entry)
            weights_by_source.setdefault(entry.source, []).append(weight)

        if not candidates_by_source:
            return None
        if self.source_sampling_strategy == "source_first":
            source_candidates = []
            source_weights = []
            for source, candidates in candidates_by_source.items():
                source_weight = self.source_weights.get(source, 0.0)
                if source_weight <= 0 or not candidates:
                    continue
                source_candidates.append(source)
                source_weights.append(source_weight)
            if not source_candidates:
                return None
            source = self.rng.choices(source_candidates, weights=source_weights, k=1)[0]
            candidates = candidates_by_source[source]
            weights = weights_by_source[source]
        else:
            candidates = []
            weights = []
            for source, source_candidates in candidates_by_source.items():
                candidates.extend(source_candidates)
                weights.extend(weights_by_source[source])

        if prefer_longest_below_max and max_prefix_len is not None:
            best_len = max(entry.prefix_len for entry in candidates)
            filtered = [(entry, weight) for entry, weight in zip(candidates, weights) if entry.prefix_len == best_len]
            candidates = [entry for entry, _ in filtered]
            weights = [weight for _, weight in filtered]

        entry = self.rng.choices(candidates, weights=weights, k=1)[0]
        self.last_sample_stats = {
            "source": entry.source,
            "task_key": entry.task_key,
            "prefix_len": entry.prefix_len,
            "age": max(0, global_step - entry.insert_step),
            "idle_steps": max(0, global_step - entry.last_used_step),
            "use_count_before": entry.use_count,
            "verified": entry.verified,
            "replay_fail_count": entry.replay_fail_count,
            "success_ema": entry.success_ema,
        }
        entry.last_used_step = global_step
        entry.use_count += 1
        self._remove_after_sample_if_needed(entry)
        return entry

    def mark_replay_result(self, prefix_id: Optional[str], *, success: bool) -> None:
        if not prefix_id or prefix_id not in self.entries:
            return
        entry = self.entries[prefix_id]
        if success:
            entry.verified = True
        else:
            entry.replay_fail_count += 1

    def update_success(self, prefix_id: Optional[str], *, success: bool, alpha: float = 0.2) -> None:
        if not prefix_id or prefix_id not in self.entries:
            return
        entry = self.entries[prefix_id]
        value = 1.0 if success else 0.0
        entry.success_ema = value if entry.use_count <= 1 else (1 - alpha) * entry.success_ema + alpha * value

    def add_trajectory_prefixes(
        self,
        *,
        source: str,
        env_name: str,
        task_key: str,
        reset_kwargs: Dict[str, Any],
        actions: List[str],
        global_step: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        if source not in {STUDENT_SOURCE, ASSISTED_SOURCE, TEACHER_SOURCE}:
            raise ValueError(f"Unsupported VMPR prefix source: {source}")
        if not actions:
            return 0

        base_metadata = dict(metadata or {})
        base_metadata.setdefault("full_actions", list(actions))
        base_metadata.setdefault("path_source", source)

        added = 0
        self.last_add_duplicate_count = 0
        for boundary in self._boundaries_for_source(actions, source):
            prefix_actions = actions[:boundary]
            entry_metadata = dict(base_metadata)
            entry_metadata["remaining_actions"] = list(actions[boundary:])
            entry = PrefixEntry(
                prefix_id=str(uuid.uuid4()),
                source=source,
                env_name=env_name,
                task_key=task_key,
                reset_kwargs=dict(reset_kwargs or {}),
                prefix_actions=list(prefix_actions),
                boundary_type=f"step_{boundary}",
                prefix_len=len(prefix_actions),
                insert_step=global_step,
                last_used_step=global_step,
                metadata=entry_metadata,
            )
            signature = self._entry_signature(entry)
            if signature in self._seen_prefix_signatures:
                self.last_add_duplicate_count += 1
                continue
            self.entries[entry.prefix_id] = entry
            self._seen_prefix_signatures.add(signature)
            added += 1

        self._trim()
        return added

    @staticmethod
    def _entry_signature(entry: PrefixEntry) -> tuple:
        full_actions = (entry.metadata or {}).get("full_actions") or entry.prefix_actions
        normalized_actions = tuple(" ".join(str(action).lower().split()) for action in full_actions)
        return entry.source, str(entry.task_key), normalized_actions, int(entry.prefix_len)

    def _boundaries_for(self, actions: List[str]) -> List[int]:
        max_prefix_len = max(0, len(actions) - 1)
        if max_prefix_len <= 0:
            return []
        if not self.step_boundaries:
            return [max_prefix_len]
        return [boundary for boundary in self.step_boundaries if boundary <= max_prefix_len]

    def _boundaries_for_source(self, actions: List[str], source: str) -> List[int]:
        boundaries = self._boundaries_for(actions)
        if source == STUDENT_SOURCE and self.max_student_prefix_length_ratio > 0:
            max_student_prefix_len = max(0, int(len(actions) * self.max_student_prefix_length_ratio))
            boundaries = [boundary for boundary in boundaries if boundary <= max_student_prefix_len]
        if source == STUDENT_SOURCE and self.student_prefix_boundary_sample == "random_one" and boundaries:
            return [self.rng.choice(boundaries)]
        return boundaries

    def _trim(self) -> None:
        for source, limit in self.source_limits.items():
            self._trim_source(source, limit)
        if self.max_buffer_size <= 0 or len(self.entries) <= self.max_buffer_size:
            return
        overflow = len(self.entries) - self.max_buffer_size
        if overflow <= 0:
            return
        if self.preserve_teacher_seed:
            removable = [entry for entry in self.entries.values() if entry.source != TEACHER_SOURCE]
            drop = self._oldest_entries(removable, min(overflow, len(removable)))
            for entry in drop:
                self.entries.pop(entry.prefix_id, None)
            overflow = len(self.entries) - self.max_buffer_size
            if overflow <= 0:
                return

        drop = self._oldest_entries(list(self.entries.values()), overflow)
        for entry in drop:
            self.entries.pop(entry.prefix_id, None)

    def _trim_source(self, source: str, limit: Optional[int]) -> None:
        if limit is None or limit < 0:
            return
        entries = [entry for entry in self.entries.values() if entry.source == source]
        if len(entries) <= limit:
            return
        drop_count = len(entries) - limit
        for entry in self._oldest_entries(entries, drop_count):
            self.entries.pop(entry.prefix_id, None)

    @staticmethod
    def _oldest_entries(entries: List[PrefixEntry], count: int) -> List[PrefixEntry]:
        if count <= 0:
            return []
        ordered = sorted(
            entries,
            key=lambda entry: (
                entry.insert_step,
                entry.last_used_step,
                -entry.use_count,
                entry.replay_fail_count,
            ),
        )
        return ordered[:count]
