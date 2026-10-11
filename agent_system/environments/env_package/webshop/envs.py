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

import gymnasium as gym
import numpy as np
import ray

from agent_system.webshop_state_utils import (
    build_webshop_oracle_reference,
    capture_webshop_state,
    load_webshop_jsonl_manifest,
    webshop_task_id,
)

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray actor. Uses the isolated HTTP service when WEBSHOP_SERVICE_FILE is set."""

    def __init__(self, seed, env_kwargs, service_url=None):
        if service_url:
            from .http_client import RemoteWebshopWorker

            self.backend = RemoteWebshopWorker(seed, env_kwargs, service_url)
        else:
            from .local_backend import LocalWebshopWorker

            self.backend = LocalWebshopWorker(seed, env_kwargs)

    def step(self, action):
        return self.backend.step(action)

    def reset(self, idx, kwargs=None):
        return self.backend.reset(idx, kwargs)

    def get_goals(self):
        return self.backend.get_goals()

    def get_available_actions(self):
        return self.backend.get_available_actions()

    def render(self, mode_for_render):
        return self.backend.render(mode_for_render)

    def close(self):
        return self.backend.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        self._rng = np.random.RandomState(seed)

        self._env_kwargs = dict(env_kwargs or {'observation_mode': 'text', 'num_products': None})

        # -------------------------- Ray actors setup --------------------------
        env_worker = ray.remote(**resources_per_worker)(WebshopWorker)
        from .http_client import configured_service_urls

        service_urls = configured_service_urls()
        self._workers = []
        for i in range(self.num_processes):
            service_url = service_urls[i % len(service_urls)] if service_urls else None
            worker = env_worker.remote(seed + (i // self.group_n), self._env_kwargs, service_url)
            self._workers.append(worker)

        # Get goals from the first worker
        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)
        self.task_ids = [
            webshop_task_id(dict(goal), goal_index=goal_index)
            for goal_index, goal in enumerate(goals)
        ]
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError('WebShop canonical goal list contains duplicate stable task ids')

        oracle_manifest_path = self._env_kwargs.get('oracle_manifest_path')
        if oracle_manifest_path:
            records = load_webshop_jsonl_manifest(oracle_manifest_path)
            if len(records) != len(goals):
                raise ValueError(
                    f'WebShop oracle manifest has {len(records)} goals but environment loaded {len(goals)}'
                )
            for goal_index, (record, task_id) in enumerate(zip(records, self.task_ids)):
                if int(record['goal_index']) != goal_index or str(record['webshop_task_id']) != task_id:
                    raise ValueError(
                        f'WebShop oracle manifest disagrees at goal_index={goal_index}: '
                        f"manifest={record.get('webshop_task_id')} environment={task_id}"
                    )
                expected_reference = build_webshop_oracle_reference(
                    dict(goals[goal_index]),
                    query_variant=str(self._env_kwargs.get('oracle_query_variant', 'full_name')),
                )
                if (
                    str(record.get('query_variant')) != str(self._env_kwargs.get('oracle_query_variant', 'full_name'))
                    or list(record.get('full_actions') or []) != list(expected_reference.get('full_actions') or [])
                    or list(record.get('reference_states') or []) != list(expected_reference.get('reference_states') or [])
                ):
                    raise ValueError(
                        f'WebShop oracle manifest path/state mismatch at goal_index={goal_index}'
                    )

        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        if not self.is_train:
            self.goal_idxs = range(500)
            fixed_goal_count = int(self._env_kwargs.get('fixed_goal_count', self.env_num) or self.env_num)
            validation_manifest_path = self._env_kwargs.get('validation_manifest_path')
            if validation_manifest_path:
                with open(os.path.expanduser(str(validation_manifest_path)), 'r', encoding='utf-8') as handle:
                    manifest = json.load(handle)
                entries = list(manifest.get('tasks') or [])
                self.fixed_goal_indices = [int(entry['goal_index']) for entry in entries]
                expected_task_ids = [str(entry['webshop_task_id']) for entry in entries]
                actual_task_ids = [self.task_ids[index] for index in self.fixed_goal_indices]
                if expected_task_ids != actual_task_ids:
                    raise ValueError('WebShop validation manifest does not match the canonical goal order')
            else:
                validation_seed = int(self._env_kwargs.get('validation_goal_seed', seed) or seed)
                validation_rng = np.random.RandomState(validation_seed)
                self.fixed_goal_indices = validation_rng.choice(
                    list(self.goal_idxs), size=fixed_goal_count, replace=False
                ).tolist()
            if len(self.fixed_goal_indices) != fixed_goal_count:
                raise ValueError(
                    f'WebShop validation manifest has {len(self.fixed_goal_indices)} tasks; '
                    f'expected {fixed_goal_count}'
                )
            if len(set(self.fixed_goal_indices)) != len(self.fixed_goal_indices):
                raise ValueError('WebShop validation manifest contains duplicate goal indices')
        else:
            self.goal_idxs = range(500, len(goals))

        print(self.goal_idxs)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self, kwargs=None):
        if kwargs is None:
            kwargs = [{} for _ in range(self.num_processes)]
        elif isinstance(kwargs, dict):
            kwargs = [dict(kwargs) for _ in range(self.num_processes)]
        else:
            kwargs = [dict(item or {}) for item in kwargs]
            if len(kwargs) != self.num_processes:
                raise ValueError(f'Expected {self.num_processes} reset kwargs, got {len(kwargs)}')

        sampled_indices = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        sampled_indices = np.repeat(sampled_indices, self.group_n).tolist()
        reset_specs = []
        for sampled_index, reset_kwargs in zip(sampled_indices, kwargs):
            goal_index = int(reset_kwargs.get('session_id', sampled_index))
            if not 0 <= goal_index < len(self.task_ids):
                raise IndexError(f'WebShop goal index {goal_index} is outside {len(self.task_ids)} goals')
            reset_specs.append(
                {
                    **reset_kwargs,
                    'session_id': goal_index,
                    'expected_webshop_task_id': self.task_ids[goal_index],
                }
            )

        futures = []
        for worker, reset_kwargs in zip(self._workers, reset_specs):
            futures.append(worker.reset.remote(int(reset_kwargs['session_id']), reset_kwargs))

        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    def validation_reset_kwargs(self, validation_item_ids):
        """Map logical validation items and replicas to fixed WebShop goals."""
        if self.is_train:
            raise ValueError('validation_reset_kwargs is only available for validation environments')
        reset_kwargs = []
        for value in validation_item_ids:
            parts = str(value).split(':', 1)
            if len(parts) != 2:
                raise ValueError(f'Invalid validation_item_id: {value!r}')
            batch_idx, item_idx = int(parts[0]), int(parts[1])
            if batch_idx != 0:
                raise ValueError(
                    'Pooled WebShop validation currently requires one val dataloader batch; '
                    f'got validation batch index {batch_idx}'
                )
            if not 0 <= item_idx < len(self.fixed_goal_indices):
                raise IndexError(
                    f'Validation item index {item_idx} is outside '
                    f'{len(self.fixed_goal_indices)} fixed WebShop goals'
                )
            goal_index = int(self.fixed_goal_indices[item_idx])
            reset_kwargs.append(
                {
                    'session_id': goal_index,
                    'expected_webshop_task_id': self.task_ids[goal_index],
                }
            )
        return reset_kwargs

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return
        workers = list(getattr(self, '_workers', []))
        if not workers:
            self._closed = True
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
