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
import os
import sys
import numpy as np
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_LEGGED_GYM_PACKAGE_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_REPO_ROOT = os.path.dirname(_LEGGED_GYM_PACKAGE_ROOT)
for _path in (_LEGGED_GYM_PACKAGE_ROOT, os.path.join(_REPO_ROOT, "rsl_rl")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Copyright (c) 2021 ETH Zurich, Nikita Rudin
from legged_gym import LEGGED_GYM_ROOT_DIR

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

if os.getenv("LEGGED_GYM_DEBUG_CUDA", "0") == "1":
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args, headless=True):
    args.headless = headless
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    save_path = os.path.join(LEGGED_GYM_ROOT_DIR)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args,log_root = save_path)

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
