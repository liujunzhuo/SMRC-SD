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

import gym
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
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
        
        env_kwargs = dict(env_kwargs)
        self.oracle_query_variant = str(env_kwargs.pop('oracle_query_variant', 'full_name') or 'full_name')
        goal_order_seed = int(env_kwargs.pop('goal_order_seed', 0) or 0)
        for manager_only_key in (
            'fixed_goal_count',
            'validation_goal_seed',
            'oracle_manifest_path',
            'validation_manifest_path',
        ):
            env_kwargs.pop(manager_only_key, None)
        # WebShop uses this seed to shuffle its global goal list.  Keeping it
        # identical across workers makes a goal index a stable task identity;
        # rollout/task sampling randomness remains in the vector env manager.
        env_kwargs['seed'] = goal_order_seed
        self.env = gym.make('WebAgentTextEnv-v0', **env_kwargs)
        self.session_id = None
        self.task_id = None
        self.oracle_metadata = {}

    def _refresh_session_context(self):
        self.session_id = str(self.env.session)
        goal = dict(self.env.server.user_sessions[self.env.session].get('goal') or {})
        goal_index = int(self.session_id) if self.session_id.isdigit() else None
        self.task_id = webshop_task_id(goal, goal_index=goal_index)
        oracle = build_webshop_oracle_reference(goal, query_variant=self.oracle_query_variant)
        oracle_score = None
        try:
            from web_agent_site.engine.goal import get_reward

            target_asin = str(goal.get('asin') or '')
            oracle_score = float(
                get_reward(
                    self.env.server.product_item_dict[target_asin],
                    goal,
                    self.env.server.product_prices[target_asin],
                    dict(goal.get('goal_options') or {}),
                )
            )
        except (KeyError, TypeError, ValueError):
            oracle_score = None
        self.oracle_metadata = {
            'full_actions': list(oracle.get('full_actions') or []),
            'path_source': 'webshop_goal_oracle',
            'webshop_task_id': self.task_id,
            'webshop_instruction_text': str(goal.get('instruction_text') or ''),
            'webshop_reference_states': list(oracle.get('reference_states') or []),
            'webshop_query_variant': oracle.get('query_variant'),
            'webshop_target_asin': oracle.get('target_asin'),
            'webshop_target_options': dict(oracle.get('target_options') or {}),
            'webshop_oracle_expected_score': oracle_score,
        }
    
    def step(self, action):
        """Execute a step in the environment"""
        state_before = capture_webshop_state(self.env)
        session_id = self.session_id or str(self.env.session)
        oracle_metadata = dict(self.oracle_metadata)
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = reward
        info['executed_action'] = action
        info['session_id'] = session_id
        info['webshop_task_id'] = self.task_id
        info['task_key'] = self.task_id
        info['webshop_state_before'] = state_before
        info['webshop_state_after'] = None if done else capture_webshop_state(self.env)
        info['webshop_oracle_metadata'] = oracle_metadata

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        return obs, reward, done, info
    
    def reset(self, idx, kwargs=None):
        """Reset the environment, optionally replaying a VMPR action prefix."""
        kwargs = kwargs or {}
        session = kwargs.get('session_id', idx)
        if isinstance(session, str) and session.isdigit():
            session = int(session)
        obs, info = self.env.reset(session=session)
        self._refresh_session_context()
        expected_task_id = str(kwargs.get('expected_webshop_task_id') or '')
        if expected_task_id and expected_task_id != self.task_id:
            raise RuntimeError(
                f'WebShop goal-order mismatch for session={session}: '
                f'expected {expected_task_id}, got {self.task_id}'
            )
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False
        info['session_id'] = self.session_id
        info['webshop_task_id'] = self.task_id
        info['task_key'] = self.task_id
        info['webshop_state'] = capture_webshop_state(self.env)
        history = []
        replay_success = True
        replay_error = None
        for action in list(kwargs.get('prefix_actions') or []):
            history.append({'text_obs': obs, 'action': action})
            obs, _, done, step_info = self.step(action)
            info = dict(step_info or {})
            if done:
                replay_success = False
                replay_error = 'VMPR prefix reached a terminal state before suffix rollout'
                obs, info = self.env.reset(session=idx)
                self._refresh_session_context()
                info = dict(info or {})
                info['available_actions'] = self.env.get_available_actions()
                info['won'] = False
                history = []
                break
        info['vmpr_replay_success'] = replay_success
        info['vmpr_replay_error'] = replay_error
        info['vmpr_replay_history'] = history
        info['vmpr_prefix_id'] = kwargs.get('vmpr_prefix_id') if replay_success else None
        info['vmpr_failed_prefix_id'] = None if replay_success else kwargs.get('vmpr_prefix_id')
        info['vmpr_source'] = (kwargs.get('vmpr_source') or 'full_start') if replay_success else 'full_start'
        info['vmpr_prefix_len'] = len(history) if replay_success else 0
        supplied_metadata = dict(kwargs.get('vmpr_metadata') or {}) if replay_success else {}
        info['vmpr_metadata'] = {**self.oracle_metadata, **supplied_metadata}
        info['session_id'] = self.session_id
        info['webshop_task_id'] = self.task_id
        info['task_key'] = self.task_id
        info['webshop_state'] = capture_webshop_state(self.env)
        return obs, info
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals
    
    def close(self):
        """Close the environment"""
        self.env.close()


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
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), self._env_kwargs)
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
