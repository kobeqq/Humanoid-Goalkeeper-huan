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

import numpy as np

if not hasattr(np, "float"):
    np.float = float

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .base.legged_robot_move_amp import LeggedRobotMoveAmp
from .base.legged_robot_move_amp_2d import LeggedRobotMoveAmp2D
from .base.legged_robot_k1_loco_amp import LeggedRobotK1LocoAmp
from .base.legged_robot_omni_move_amp import LeggedRobotOmniMoveAmp
from .g1.g1_29_config import G129Cfg, G129CfgPPO
from .k1.k1_22_config import K122Cfg, K122CfgPPO
from .g1_loco_13.g1_loco_13_config import G1LOCO13Cfg, G1LOCO13CfgPPO
from .k1.k1_move_amp_config import K1MoveAmpCfg, K1MoveAmpCfgPPO
from .k1.k1_move_amp_2d_config import K1MoveAmp2DCfg, K1MoveAmp2DCfgPPO
from .k1.k1_loco_amp_config import K1LocoAmpCfg, K1LocoAmpCfgPPO
from .k1.k1_loco_amp_full_config import K1LocoAmpFullCfg, K1LocoAmpFullCfgPPO
from .k1.k1_omni_move_amp_config import K1OmniMoveAmpCfg, K1OmniMoveAmpCfgPPO

import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "29", LeggedRobot, G129Cfg(), G129CfgPPO() )  
task_registry.register( "g1_loco_amp13", LeggedRobot, G1LOCO13Cfg(), G1LOCO13CfgPPO() )

task_registry.register( "k1", LeggedRobot, K122Cfg(), K122CfgPPO() )
task_registry.register( "k1_move_amp", LeggedRobotMoveAmp, K1MoveAmpCfg(), K1MoveAmpCfgPPO() )
task_registry.register( "k1_move_amp_2d", LeggedRobotMoveAmp2D, K1MoveAmp2DCfg(), K1MoveAmp2DCfgPPO() )
task_registry.register( "k1_loco_amp", LeggedRobotK1LocoAmp, K1LocoAmpCfg(), K1LocoAmpCfgPPO() )
task_registry.register( "k1_loco_amp_full", LeggedRobotK1LocoAmp, K1LocoAmpFullCfg(), K1LocoAmpFullCfgPPO() )
task_registry.register( "k1_omni_move_amp", LeggedRobotOmniMoveAmp, K1OmniMoveAmpCfg(), K1OmniMoveAmpCfgPPO() )
