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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class G1RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 2048
        num_observations = 235
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.0  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = "plane"
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_dict = {
            "smooth slope": 0.2,
            "rough slope": 0.2,
            "rough stairs up": 0.1, 
            "rough stairs down": 0.1, 
            "discrete": 0.1, 
            "stepping stones": 0.1,
            "gaps": 0.2, 
            }
        max_init_terrain_level = 6
        terrain_proportions = list(terrain_dict.values())
    
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           "left_hip_yaw_joint": 0.0,   
           "left_hip_roll_joint": 0.0,               
           "left_hip_pitch_joint": -0.1,         
           "left_knee_joint": 0.3,       
           "left_ankle_pitch_joint": -0.2,     
           "left_ankle_roll_joint": 0.0,     
           "right_hip_yaw_joint": 0.0, 
           "right_hip_roll_joint": 0.0, 
           "right_hip_pitch_joint": -0.1,                                       
           "right_knee_joint": 0.3,                                             
           "right_ankle_pitch_joint": -0.2,                              
           "right_ankle_roll_joint": 0.0,       
           "torso_joint": 0.0,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = "P"
          # PD Drive parameters:
        stiffness = {"hip_yaw": 100,
                     "hip_roll": 100,
                     "hip_pitch": 100,
                     "knee": 150,
                     "ankle": 40,
                     }  # [N*m/rad]
        damping = {  "hip_yaw": 2,
                     "hip_roll": 2,
                     "hip_pitch": 2,
                     "knee": 4,
                     "ankle": 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/urdf/g1_12dof.urdf"
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        
        class scales:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            torques = -0.0002
            dof_acc = -2.5e-7
            feet_air_time = 1.0
            collision = -1.0
            action_rate = -0.01
            dof_pos_limits = -10.0
            

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5e-4
        schedule = "adaptive" # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 32 # per iteration
        max_iterations = 100000 # number of policy updates
        
        run_name = "rough_g1"
        experiment_name = "rough_g1"