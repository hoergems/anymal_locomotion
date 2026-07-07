# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import torch
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip

def randomize_target_velocity(env, env_ids: torch.Tensor | None):    
    env.randomize_target_velocity(env_ids=env_ids)

@configclass
class EventCfg:
    """Configuration of environment events."""

    # Apply a new random command (desired base velocity)
    # every 2 seconds
    set_target_velocity = EventTerm(
        func=randomize_target_velocity,
        mode="interval",
        interval_range_s=(2.0, 2.0),        
    )

@configclass
class AnymalCommandTrackingEnvCfg(DirectRLEnvCfg):
    # Simulation setup
    # Physics integration time step (in seconds)
    dt = 0.005
    # One environment step consists of 4 physics integration steps
    decimation = 4

    sim: SimulationCfg = SimulationCfg(
        dt=dt,
        render_interval=decimation,

        # Default physics material setting for rigid bodies
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Configure flat terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Configuration of the simulated scene and parallel environments
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # Configuration of the ANYmal C robot
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Configuration of a contact sensor attached to the robot
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # Configuration of environment events (command randomization in this case)
    events: EventCfg = EventCfg()

    # RL settings
    # Maximum duration of an episode
    episode_length_s = 20.0
    # Size of the action space
    action_space = 12
    # Size of the observation space
    observation_space = 48
    # Size of the privileged state space (0 = not used)
    state_space = 0

    # Reward term scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -5.0

    # Constant action scaling factor
    action_scale = 0.5
