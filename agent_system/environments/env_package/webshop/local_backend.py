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

"""CPU-only WebShop worker used by both in-process actors and the isolated service."""

import os

import gym

from agent_system.webshop_state_utils import (
    build_webshop_oracle_reference,
    capture_webshop_state,
    webshop_task_id,
)

class LocalWebshopWorker:
    """In-process WebShop simulator. No Ray or inference dependencies."""
    
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


