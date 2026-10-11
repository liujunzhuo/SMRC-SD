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

import os
import yaml
import gymnasium as gym
import numpy as np
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =


def select_eval_game_files(game_files, env_num, seed):
    """Select a deterministic validation subset without replacement."""
    sorted_game_files = sorted(game_files)
    if env_num > len(sorted_game_files):
        raise ValueError(f"Requested {env_num} ALFWorld validation games, but split only contains {len(sorted_game_files)}")
    rng = np.random.RandomState(seed)
    selected_indices = rng.permutation(len(sorted_game_files))[:env_num]
    return [sorted_game_files[idx] for idx in selected_indices]


def load_fixed_game_files(value):
    """Load an optional fixed ALFWorld gamefile list from a path or sequence."""
    if value in (None, "", "null", "None"):
        return None
    if isinstance(value, (list, tuple)):
        game_files = [str(item).strip() for item in value if str(item).strip()]
    else:
        path = os.path.expanduser(os.path.expandvars(str(value)))
        with open(path, encoding="utf-8") as f:
            game_files = [line.strip() for line in f if line.strip()]
    return game_files or None


def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    # Keep the heavyweight vision stack out of text-only ALFWorld Ray actors.
    # AlfredTWEnv never calls this function, while eager module imports make
    # every resident environment process load torch and torchvision.
    import torch
    import torchvision.transforms as T

    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, seed, base_env=None):
        # Fixed-game validation always constructs a checkpoint TextWorld env on
        # reset.  Do not also retain an unused base batched environment in every
        # validation actor.
        self.env = base_env.init_env(batch_size=1) if base_env is not None else None
        if self.env is not None:
            self.env.seed(seed)
        self.active_env = self.env
        self._active_single_env = False
        self._checkpoint_env = None

    def _wrap_single_info(self, info, obs, reward=None, done=None):
        wrapped = {key: [value] for key, value in dict(info or {}).items()}
        wrapped["observation_text"] = [obs]
        if reward is not None:
            wrapped["vmpr_replay_reward"] = [reward]
        if done is not None:
            wrapped["vmpr_replay_done"] = [done]
        return wrapped

    def _close_checkpoint_env(self):
        if self._checkpoint_env is not None:
            try:
                self._checkpoint_env.close()
            except Exception:
                pass
        self._checkpoint_env = None

    def _make_checkpoint_env(self, gamefile):
        import textworld
        import textworld.gym
        from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env_id = textworld.gym.register_game(gamefile, request_infos, wrappers=[AlfredDemangler(), AlfredInfos])
        return textworld.gym.make(env_id)

    def step(self, action):
        """Execute a step in the environment"""
        if self._active_single_env:
            obs, score, done, info = self.active_env.step(action)
            info = self._wrap_single_info(info, obs)
            return [obs], [score], [done], info

        actions = [action]
        obs, scores, dones, infos = self.env.step(actions)
        infos["observation_text"] = obs
        return obs, scores, dones, infos

    def reset(self, kwargs=None):
        """Reset the environment, optionally replaying a VMPR action prefix."""
        kwargs = kwargs or {}
        gamefile = kwargs.get("gamefile")
        prefix_actions = list(kwargs.get("prefix_actions") or [])
        if gamefile:
            try:
                self._close_checkpoint_env()
                self._checkpoint_env = self._make_checkpoint_env(gamefile)
                self.active_env = self._checkpoint_env
                self._active_single_env = True
                obs, info = self.active_env.reset()
                task_start = obs.find("Your task is to: ")
                task_description = obs[task_start + len("Your task is to: "):].strip() if task_start != -1 else ""
                history = []
                done = False
                reward = 0.0
                for action in prefix_actions:
                    history.append({"text_obs": obs, "action": action})
                    obs, reward, done, info = self.active_env.step(action)
                    if done:
                        raise RuntimeError("VMPR prefix reached a terminal state before suffix rollout")
                info = self._wrap_single_info(info, obs, reward=reward, done=done)
                info["vmpr_replay_success"] = [True]
                info["vmpr_replay_history"] = [history]
                info["vmpr_prefix_id"] = [kwargs.get("vmpr_prefix_id")]
                info["vmpr_source"] = [kwargs.get("vmpr_source") or "full_start"]
                info["vmpr_prefix_len"] = [len(prefix_actions)]
                info["vmpr_metadata"] = [kwargs.get("vmpr_metadata") or {}]
                info["vmpr_task_description"] = [task_description]
                info["extra.gamefile"] = [gamefile]
                return [obs], info
            except Exception as exc:
                if kwargs.get("strict_gamefile", False):
                    raise RuntimeError(f"Failed to reset requested ALFWorld gamefile {gamefile}: {exc}") from exc
                self._close_checkpoint_env()
                self.active_env = self.env
                self._active_single_env = False
                if self.env is None:
                    raise RuntimeError("No fallback ALFWorld base environment is available for this fixed-game worker") from exc
                obs, infos = self.env.reset()
                infos["observation_text"] = obs
                infos["vmpr_replay_success"] = [False]
                infos["vmpr_replay_error"] = [str(exc)]
                infos["vmpr_failed_prefix_id"] = [kwargs.get("vmpr_prefix_id")]
                infos["vmpr_replay_history"] = [[]]
                infos["vmpr_prefix_id"] = [None]
                infos["vmpr_source"] = ["full_start"]
                infos["vmpr_prefix_len"] = [0]
                infos["vmpr_metadata"] = [{}]
                return obs, infos

        self._close_checkpoint_env()
        self.active_env = self.env
        self._active_single_env = False
        if self.env is None:
            raise RuntimeError("Fixed-game ALFWorld validation workers require reset kwargs with a gamefile")
        obs, infos = self.env.reset()
        infos["observation_text"] = obs
        infos["vmpr_replay_success"] = [True]
        infos["vmpr_replay_history"] = [[]]
        infos["vmpr_prefix_id"] = [None]
        infos["vmpr_source"] = ["full_start"]
        infos["vmpr_prefix_len"] = [0]
        infos["vmpr_metadata"] = [{}]
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs=None):
        super().__init__()
        env_kwargs = env_kwargs or {}
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n
        self.is_train = is_train
        self.fixed_game_files = None
        train_game_files = load_fixed_game_files(env_kwargs.get("train_game_files"))
        if is_train and train_game_files is not None:
            if env_num > len(train_game_files):
                raise ValueError(f"Requested {env_num} ALFWorld train games, but fixed list only contains {len(train_game_files)}")
            self.fixed_game_files = list(train_game_files[:env_num])
        elif not is_train:
            fixed_game_file_count = int(env_kwargs.get("fixed_game_file_count", env_num) or env_num)
            self.fixed_game_files = select_eval_game_files(base_env.game_files, fixed_game_file_count, seed)

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), base_env if is_train else None)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self, kwargs=None):
        """
        Send reset commands to workers. kwargs may contain per-env VMPR prefix replay specs.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        if kwargs is None:
            if self.fixed_game_files is None:
                kwargs = [None] * self.num_processes
            else:
                kwargs = [
                    {
                        "gamefile": self.fixed_game_files[i // self.group_n],
                        "strict_gamefile": True,
                    }
                    for i in range(self.num_processes)
                ]
        elif isinstance(kwargs, dict):
            kwargs = [kwargs] * self.num_processes
        else:
            kwargs = list(kwargs)
            if len(kwargs) != self.num_processes:
                raise ValueError(f"Expected {self.num_processes} reset kwargs, got {len(kwargs)}")

        futures = []
        for worker, reset_kwargs in zip(self.workers, kwargs):
            futures.append(worker.reset.remote(reset_kwargs))

        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def validation_reset_kwargs(self, validation_item_ids):
        """Map logical validation items to deterministic gamefiles.

        Validation replicas share an item id and therefore reset to the same
        gamefile.  The current trainer creates a single validation dataloader
        batch; fail loudly instead of silently repeating games if that changes.
        """
        if self.is_train or self.fixed_game_files is None:
            raise ValueError("validation_reset_kwargs is only available for fixed ALFWorld validation environments")
        reset_kwargs = []
        for value in validation_item_ids:
            parts = str(value).split(":", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid validation_item_id: {value!r}")
            batch_idx, item_idx = int(parts[0]), int(parts[1])
            if batch_idx != 0:
                raise ValueError(
                    "Pooled ALFWorld validation currently requires one val dataloader batch; "
                    f"got validation batch index {batch_idx}"
                )
            if not 0 <= item_idx < len(self.fixed_game_files):
                raise IndexError(f"Validation item index {item_idx} is outside {len(self.fixed_game_files)} fixed games")
            reset_kwargs.append({"gamefile": self.fixed_game_files[item_idx], "strict_gamefile": True})
        return reset_kwargs

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs=None):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)
