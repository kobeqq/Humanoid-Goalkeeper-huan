# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

try:
    from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
except ModuleNotFoundError:
    class TensorboardSummaryWriter:
        def __init__(self, *args, **kwargs):
            print("[rsl_rl] tensorboard is not installed; scalar logs will be skipped.")

        def add_scalar(self, *args, **kwargs):
            pass

        def save_file(self, *args, **kwargs):
            pass

        def flush(self):
            pass

        def close(self):
            pass

import torch

import rsl_rl
from rsl_rl.algorithms import HIMPPO
from rsl_rl.modules import ActorCritic

from rsl_rl.env import VecEnv
from rsl_rl.utils import store_code_state
from copy import copy, deepcopy
import warnings
from rsl_rl.modules import AMP
from rsl_rl.utils.utils import Normalizer


class _MultiMotionBuffer:
    def __init__(self, motion_buffers, probs=None, names=None, command_motion_names=None):
        self.motion_buffers = list(motion_buffers)
        self.names = list(names) if names is not None else [str(i) for i in range(len(self.motion_buffers))]
        self.name_lowers = [name.lower() for name in self.names]
        if len(self.motion_buffers) == 0:
            raise RuntimeError("No AMP motion buffers were provided.")
        if probs is None:
            self.probs = torch.ones(len(self.motion_buffers), dtype=torch.float)
        else:
            self.probs = torch.as_tensor(probs, dtype=torch.float).detach().cpu()
        if self.probs.numel() != len(self.motion_buffers):
            raise RuntimeError(
                f"AMP motion prob length mismatch: probs={self.probs.numel()} buffers={len(self.motion_buffers)}"
            )
        if torch.sum(self.probs) <= 0:
            raise RuntimeError("AMP motion sampling probabilities sum to zero.")
        self.probs = self.probs / self.probs.sum()
        self.command_motion_names = list(command_motion_names) if command_motion_names is not None else None
        self.command_buffer_indices = None
        if self.command_motion_names is not None:
            self.command_buffer_indices = [
                self._resolve_motion_indices(name) for name in self.command_motion_names
            ]

    def _resolve_motion_indices(self, motion_name):
        if motion_name is None:
            return []
        names = motion_name if isinstance(motion_name, (list, tuple)) else [motion_name]
        resolved = []
        for name in names:
            name_lower = str(name).lower()
            exact = [i for i, candidate in enumerate(self.name_lowers) if candidate == name_lower]
            if exact:
                for idx in exact:
                    if idx not in resolved:
                        resolved.append(idx)
                continue
            partial = [
                i for i, candidate in enumerate(self.name_lowers)
                if name_lower in candidate or candidate in name_lower
            ]
            for idx in partial:
                if idx not in resolved:
                    resolved.append(idx)
        return resolved

    def _sample_from_indices(self, batch_size, buffer_indices):
        if len(buffer_indices) == 0:
            return self._get_unconditional_expert_obs(batch_size)
        if len(buffer_indices) == 1:
            return self.motion_buffers[buffer_indices[0]].get_expert_obs(batch_size=batch_size)

        local_probs = self.probs[buffer_indices]
        local_probs = local_probs / local_probs.sum().clamp(min=1e-6)
        selected = torch.multinomial(local_probs, batch_size, replacement=True)
        counts = torch.bincount(selected, minlength=len(buffer_indices))
        result = None
        for local_idx, count in enumerate(counts.tolist()):
            if count <= 0:
                continue
            rows = (selected == local_idx).nonzero(as_tuple=False).flatten()
            sample = self.motion_buffers[buffer_indices[local_idx]].get_expert_obs(batch_size=count)
            if result is None:
                result = sample.new_empty(batch_size, sample.shape[-1])
            result[rows.to(sample.device)] = sample
        return result

    def _get_unconditional_expert_obs(self, batch_size):
        if len(self.motion_buffers) == 1:
            return self.motion_buffers[0].get_expert_obs(batch_size=batch_size)
        ids = torch.multinomial(self.probs, batch_size, replacement=True)
        counts = torch.bincount(ids, minlength=len(self.motion_buffers))
        result = None
        for idx, count in enumerate(counts.tolist()):
            if count > 0:
                rows = (ids == idx).nonzero(as_tuple=False).flatten()
                sample = self.motion_buffers[idx].get_expert_obs(batch_size=count)
                if result is None:
                    result = sample.new_empty(batch_size, sample.shape[-1])
                result[rows.to(sample.device)] = sample
        return result

    def get_expert_obs(self, batch_size, motion_ids=None):
        if motion_ids is None or self.command_buffer_indices is None:
            return self._get_unconditional_expert_obs(batch_size)

        motion_ids = motion_ids.detach().view(-1).long().cpu()
        if motion_ids.numel() != batch_size:
            return self._get_unconditional_expert_obs(batch_size)

        result = None
        for command_id in torch.unique(motion_ids).tolist():
            rows = (motion_ids == command_id).nonzero(as_tuple=False).flatten()
            count = rows.numel()
            if count == 0:
                continue
            if 0 <= command_id < len(self.command_buffer_indices):
                sample = self._sample_from_indices(count, self.command_buffer_indices[command_id])
            else:
                sample = self._get_unconditional_expert_obs(count)
            if result is None:
                result = sample.new_empty(batch_size, sample.shape[-1])
            result[rows.to(sample.device)] = sample
        return result


class HIMOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_one_step_obs
        
        self.num_actor_obs = self.env.num_obs
        
        self.num_critic_obs = num_critic_obs
        self.actor_history_length = self.env.actor_history_length
        
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( 
                                                        self.num_actor_obs,
                                                        self.num_critic_obs,
                                                        self.env.num_one_step_obs,
                                                        self.actor_history_length,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)

        self.amp_cfg = train_cfg["amp"]
        self.amp_coef = self.amp_cfg['amp_coef']
        self.amp_scale = self.amp_cfg.get("amp_scale", self.amp_coef)
        self.amp_reward_mode = self.amp_cfg.get("reward_mode", "mixture")
        self.adaptive_amp_scale = self.amp_cfg.get("adaptive_amp_scale", False)
        self.amp_target_fraction = self.amp_cfg.get("amp_target_fraction", self.amp_coef)
        self.amp_scale_min = self.amp_cfg.get("amp_scale_min", 0.0)
        self.amp_scale_max = self.amp_cfg.get("amp_scale_max", float("inf"))
        self.amp_scale_ema_alpha = self.amp_cfg.get("amp_scale_ema_alpha", 1.0)
        self.enable_discriminator = self.amp_cfg.get('enable_discriminator', True)
        self.amp_log_network = self.amp_cfg.get("log_network", False)
        if self.enable_discriminator:
            amp = AMP(self.amp_cfg['num_obs'], self.amp_cfg['amp_coef'], device=self.device).to(self.device)
            amp_normalizer = Normalizer(self.amp_cfg['num_obs'])
            motions = getattr(self.env, "motions", {})
            if not motions:
                raise RuntimeError(
                    "AMP discriminator is enabled, but env.motions is empty. "
                    "Provide a motion dataset or set cfg.amp.enable_discriminator=False."
                )
            motion_buffers = getattr(self.env, "amp_motion_buffers", None)
            motion_probs = getattr(self.env, "amp_motion_probs", None)
            motion_names = getattr(self.env, "amp_motion_names", None)
            if motion_buffers is None:
                motion_buffers = list(motions.values())
                motion_probs = None
                motion_names = list(motions.keys())
            command_motion_names = getattr(self.env, "amp_command_motion_names", None)
            motion_buffer = _MultiMotionBuffer(
                motion_buffers,
                probs=motion_probs,
                names=motion_names,
                command_motion_names=command_motion_names,
            )
            print(
                "[AMP] discriminator enabled: "
                f"obs_dim={self.amp_cfg['num_obs']}, "
                f"reward_mode={self.amp_reward_mode}, "
                f"amp_coef={self.amp_coef}, amp_scale={self.amp_scale}, "
                f"adaptive_amp_scale={self.adaptive_amp_scale}, "
                f"num_motion_buffers={len(motion_buffer.motion_buffers)}, "
                f"command_conditioned={command_motion_names is not None}"
            )
            print(f"[AMP] motion buffers: {motion_buffer.names}")
            print(f"[AMP] motion probs: {motion_buffer.probs.tolist()}")
            if command_motion_names is not None:
                command_names = getattr(self.env, "amp_command_names", None)
                if command_names is None:
                    command_names = [str(i) for i in range(len(command_motion_names))]
                print(f"[AMP] command motion map: {dict(zip(command_names, command_motion_names))}")
                for motion_id, (command_name, motion_name) in enumerate(
                    zip(command_names, command_motion_names)
                ):
                    buffer_indices = motion_buffer.command_buffer_indices[motion_id]
                    print(
                        "[AMP] command mapping: "
                        f"command={command_name}, motion_id={motion_id}, "
                        f"motion_name={motion_name}, buffer_indices={buffer_indices}"
                    )
                validation_ids = torch.arange(len(command_names), device=self.device)
                validation_obs = motion_buffer.get_expert_obs(
                    batch_size=len(command_names), motion_ids=validation_ids
                )
                expected_amp_dim = int(self.amp_cfg['num_obs'])
                if validation_obs.shape != (len(command_names), expected_amp_dim):
                    raise RuntimeError(
                        "AMP expert observation shape mismatch: "
                        f"got {tuple(validation_obs.shape)}, "
                        f"expected {(len(command_names), expected_amp_dim)}"
                    )
                if not torch.isfinite(validation_obs).all():
                    raise RuntimeError("AMP expert observation validation found non-finite values.")
                print(f"[AMP] expert observation validation: shape={tuple(validation_obs.shape)}")
            if self.amp_log_network:
                print(f"[AMP] discriminator network:\n{amp}")
        else:
            amp = None
            amp_normalizer = None
            motion_buffer = None
            if self.amp_cfg.get("verbose", False):
                print("[AMP] discriminator disabled")


        alg_class = eval(self.cfg["algorithm_class_name"]) # HIMPPO
        self.alg: HIMPPO = alg_class(
            actor_critic,
            amp=amp,
            amp_normalizer=amp_normalizer,
            motion_buffer=motion_buffer,
            enable_discriminator=self.enable_discriminator,
            device=self.device,
            **self.alg_cfg,
        )
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions],  [self.env.num_amp_obs])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

        _, _ = self.env.reset()



    
    def _get_amp_motion_ids(self):
        if hasattr(self.env, "get_amp_motion_ids"):
            motion_ids = self.env.get_amp_motion_ids()
            if motion_ids is None:
                return None
            return motion_ids.to(self.device).clone()
        motion_ids = getattr(self.env, "command_type_ids", None)
        if motion_ids is None:
            return None
        return motion_ids.to(self.device).clone()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.logger_type = self.cfg.get("logger", "wandb")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")
            
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        if self.enable_discriminator:
            amp_state = self.env.get_amp_observations().to(self.device)
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        raw_rewbuffer = deque(maxlen=100)
        amp_rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_raw_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_amp_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            rollout_raw_reward_mean = 0.0
            rollout_amp_reward_mean = 0.0
            rollout_task_contribution_mean = 0.0
            rollout_amp_contribution_mean = 0.0
            rollout_amp_normalizer_clip_ratio = 0.0
            command_names = getattr(self.env, "amp_command_names", None) or []
            rollout_motion_id_counts = (
                torch.zeros(
                    len(command_names),
                    dtype=torch.long,
                    device=self.device,
                )
                if self.enable_discriminator and len(command_names) > 0 else None
            )
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    amp_motion_ids = self._get_amp_motion_ids() if self.enable_discriminator else None
                    if rollout_motion_id_counts is not None and amp_motion_ids is not None:
                        rollout_motion_id_counts += torch.bincount(
                            amp_motion_ids.view(-1).long(),
                            minlength=rollout_motion_id_counts.numel(),
                        )[:rollout_motion_id_counts.numel()]
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, raw_rewards, dones, infos, termination_ids, termination_privileged_obs = self.env.step(actions)

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, raw_rewards, dones = obs.to(self.device), critic_obs.to(self.device), raw_rewards.to(self.device), dones.to(self.device)
                    termination_ids = termination_ids.to(self.device)
                    termination_privileged_obs = termination_privileged_obs.to(self.device)

                    if self.enable_discriminator:
                        old_amp_state = amp_state
                        amp_state = self.env.get_amp_observations().to(self.device)
                        amp_state_ = torch.cat([old_amp_state, amp_state], dim=1).to(self.device)
                        self.alg.process_amp_state(amp_state_, amp_motion_ids)
                        amp_reward = self.alg.amp.predict_reward(
                            amp_state_,
                            normalizer=self.alg.amp_normalizer,
                        ).squeeze(1) * 0.5
                        if self.amp_reward_mode == "additive":
                            if self.adaptive_amp_scale:
                                raw_mag = raw_rewards.detach().abs().mean()
                                amp_mag = amp_reward.detach().abs().mean().clamp(min=1e-6)
                                target = min(max(float(self.amp_target_fraction), 0.0), 0.95)
                                target_scale = (target / max(1.0 - target, 1e-6)) * raw_mag / amp_mag
                                target_scale = torch.clamp(
                                    target_scale,
                                    min=float(self.amp_scale_min),
                                    max=float(self.amp_scale_max),
                                )
                                alpha = min(max(float(self.amp_scale_ema_alpha), 0.0), 1.0)
                                self.amp_scale = (1.0 - alpha) * float(self.amp_scale) + alpha * float(target_scale.item())
                            task_contribution = raw_rewards
                            amp_contribution = self.amp_scale * amp_reward
                        else:
                            task_contribution = raw_rewards * (1 - self.amp_coef)
                            amp_contribution = amp_reward * self.amp_coef
                        rewards = task_contribution + amp_contribution
                    else:
                        amp_reward = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
                        task_contribution = raw_rewards
                        amp_contribution = torch.zeros_like(raw_rewards)
                        rewards = raw_rewards

                    rollout_raw_reward_mean += raw_rewards.detach().mean().item()
                    rollout_amp_reward_mean += amp_reward.detach().mean().item()
                    rollout_task_contribution_mean += task_contribution.detach().mean().item()
                    rollout_amp_contribution_mean += amp_contribution.detach().mean().item()

                    if hasattr(self.env, "record_amp_reward"):
                        self.env.record_amp_reward(amp_reward, active_mask=(dones <= 0))

                    next_critic_obs = critic_obs.clone().detach()
                    next_critic_obs[termination_ids] = termination_privileged_obs.clone().detach()

                    self.alg.process_env_step(rewards, dones, infos, next_critic_obs)
                
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                            if len(done_ids) > 0:
                                episode_amp_reward = (
                                    cur_amp_reward_sum[done_ids] + amp_reward[done_ids]
                                ) / torch.clamp(cur_episode_length[done_ids] + 1.0, min=1.0)
                                infos['episode']['amp_reward_mean'] = torch.mean(episode_amp_reward)
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_raw_reward_sum += raw_rewards
                        cur_amp_reward_sum += amp_reward
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        raw_rewbuffer.extend(cur_raw_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        amp_rewbuffer.extend(cur_amp_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_raw_reward_sum[new_ids] = 0
                        cur_amp_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                rollout_raw_reward_mean /= self.num_steps_per_env
                rollout_amp_reward_mean /= self.num_steps_per_env
                rollout_task_contribution_mean /= self.num_steps_per_env
                rollout_amp_contribution_mean /= self.num_steps_per_env
                reward_contribution_abs_sum = (
                    abs(rollout_task_contribution_mean) + abs(rollout_amp_contribution_mean)
                )
                rollout_amp_abs_fraction = (
                    abs(rollout_amp_contribution_mean) / reward_contribution_abs_sum
                    if reward_contribution_abs_sum > 1e-12 else 0.0
                )
                if self.enable_discriminator and self.alg.amp_normalizer is not None:
                    normalized_amp_state = self.alg.amp_normalizer.normalize_torch(
                        amp_state_, self.device
                    )
                    clip_value = float(self.alg.amp_normalizer.clip_obs)
                    rollout_amp_normalizer_clip_ratio = (
                        torch.abs(normalized_amp_state) >= clip_value - 1e-6
                    ).float().mean().item()

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_est_loss, mean_region_loss, amp_loss, expert_loss, policy_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type == "wandb" and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)
            self.current_learning_iteration = it
        
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/estball', locs['mean_est_loss'], locs['it'])
        # self.writer.add_scalar('Loss/region', locs['mean_region_loss'], locs['it'])

        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Loss/amp_loss', locs['amp_loss'], locs['it'])
        self.writer.add_scalar('Loss/amp_expert_loss', locs['expert_loss'], locs['it'])
        self.writer.add_scalar('Loss/amp_policy_loss', locs['policy_loss'], locs['it'])
        self.writer.add_scalar('Train/amp_scale', self.amp_scale, locs['it'])
        self.writer.add_scalar('Train/amp_coef', self.amp_coef, locs['it'])
        self.writer.add_scalar('Train/rollout_raw_task_reward_mean', locs['rollout_raw_reward_mean'], locs['it'])
        self.writer.add_scalar('Train/rollout_raw_amp_reward_mean', locs['rollout_amp_reward_mean'], locs['it'])
        self.writer.add_scalar('Train/rollout_task_contribution_mean', locs['rollout_task_contribution_mean'], locs['it'])
        self.writer.add_scalar('Train/rollout_amp_contribution_mean', locs['rollout_amp_contribution_mean'], locs['it'])
        self.writer.add_scalar('Train/rollout_amp_abs_fraction', locs['rollout_amp_abs_fraction'], locs['it'])
        self.writer.add_scalar('Train/amp_normalizer_clip_ratio', locs['rollout_amp_normalizer_clip_ratio'], locs['it'])
        if self.alg.amp_normalizer is not None:
            amp_norm_std = self.alg.amp_normalizer.var ** 0.5
            self.writer.add_scalar(
                'Train/amp_normalizer_mean_abs',
                float(abs(self.alg.amp_normalizer.mean).mean()),
                locs['it'],
            )
            self.writer.add_scalar(
                'Train/amp_normalizer_std_min', float(amp_norm_std.min()), locs['it']
            )
            self.writer.add_scalar(
                'Train/amp_normalizer_std_max', float(amp_norm_std.max()), locs['it']
            )
        command_names = getattr(self.env, "amp_command_names", None) or []
        if locs['rollout_motion_id_counts'] is not None and len(command_names) > 0:
            motion_total = max(int(locs['rollout_motion_id_counts'].sum().item()), 1)
            for motion_id, command_name in enumerate(command_names):
                fraction = float(locs['rollout_motion_id_counts'][motion_id].item()) / motion_total
                self.writer.add_scalar(
                    f'Train/command_fraction/{motion_id}_{command_name}', fraction, locs['it']
                )
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_raw_reward', statistics.mean(locs['raw_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_amp_reward', statistics.mean(locs['amp_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
                self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimate ball loss:':>{pad}} {locs['mean_est_loss']:.4f}\n"""
                        #   f"""{'Region loss:':>{pad}} {locs['mean_region_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimate ball loss:':>{pad}} {locs['mean_est_loss']:.4f}\n"""
                        #   f"""{'Region loss:':>{pad}} {locs['mean_region_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)



    def log_vision(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/action_loss, ', locs['action_loss'], locs['it'])
        self.writer.add_scalar('Loss/est_loss', locs['est_loss'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['raw_rewbuffer']) > 0:

            self.writer.add_scalar('Train/mean_raw_reward', statistics.mean(locs['raw_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar('Train/mean_raw_reward/time', statistics.mean(locs['raw_rewbuffer']), self.tot_time)
                self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs['raw_rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Action loss:':>{pad}} {locs['action_loss']:.4f}\n"""
                          f"""{'Est loss:':>{pad}} {locs['est_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean raw_reward':>{pad}} {statistics.mean(locs['raw_rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Action loss:':>{pad}} {locs['action_loss']:.4f}\n"""
                          f"""{'Est loss:':>{pad}} {locs['est_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)


        
    def save(self, path, infos=None):
        state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration + 1,
            'infos': infos,
            }
        if self.enable_discriminator and self.alg.amp is not None:
            state_dict['amp_state_dict'] = self.alg.amp.state_dict()
        if self.alg.amp_normalizer is not None:
            state_dict['amp_normalizer'] = {
                'mean': self.alg.amp_normalizer.mean,
                'var': self.alg.amp_normalizer.var,
                'count': self.alg.amp_normalizer.count,
            }
        state_dict['amp_scale'] = self.amp_scale
        torch.save(state_dict, path)


    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])

        if self.enable_discriminator and self.alg.amp is not None:
            if 'amp_state_dict' in loaded_dict:
                self.alg.amp.load_state_dict(loaded_dict['amp_state_dict'])
            else:
                warnings.warn(
                    "Checkpoint has no AMP discriminator state; it will be randomly initialized. "
                    "Use a newly saved checkpoint before resuming AMP training.",
                    RuntimeWarning,
                )
        normalizer_state = loaded_dict.get('amp_normalizer')
        if self.alg.amp_normalizer is not None and normalizer_state is not None:
            self.alg.amp_normalizer.mean = normalizer_state['mean']
            self.alg.amp_normalizer.var = normalizer_state['var']
            self.alg.amp_normalizer.count = normalizer_state['count']
        elif self.alg.amp_normalizer is not None:
            warnings.warn(
                "Checkpoint has no AMP normalizer state; raw-observation statistics will restart.",
                RuntimeWarning,
            )
        self.amp_scale = float(loaded_dict.get('amp_scale', self.amp_scale))

        can_restore_amp_training = (
            not self.enable_discriminator
            or self.alg.amp is None
            or ('amp_state_dict' in loaded_dict and normalizer_state is not None)
        )
        if load_optimizer and can_restore_amp_training:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        elif load_optimizer:
            warnings.warn(
                "Skipped optimizer restore because this legacy checkpoint cannot restore "
                "the discriminator/normalizer that its AMP optimizer moments belong to.",
                RuntimeWarning,
            )
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_critic_policy(self, device = None):
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.evaluate

    def train_mode(self):
        self.alg.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
