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

@configclass
class AnymalNavEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0    
    decimation = 4
    action_scale = 0.5
    action_space = 12
    observation_space = 48
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
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

    goal_radius = 1.0
    goal_position = (8.0, 5.0, 0.0)

    # By default (during policy training), we don't add obstacles 
    # to the environment
    add_obstacles = True
    obstacles: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={            
            "box_1": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Box_1",
                spawn=sim_utils.CuboidCfg(
                    size=(1.0, 1.0, 1.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.2, 0.2, 0.8),
                    ),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0,
                        dynamic_friction=1.0,
                        restitution=0.0,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(2.0, 0.75, 0.5),
                ),
            ),
            "box_2": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Box_2",
                spawn=sim_utils.CuboidCfg(
                    size=(1.0, 1.0, 1.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.2, 0.2, 0.8),
                    ),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0,
                        dynamic_friction=1.0,
                        restitution=0.0,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(5.0, 3.0, 0.5),
                ),
            ),
            "box_3": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Box_3",
                spawn=sim_utils.CuboidCfg(
                    size=(1.0, 1.0, 1.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.2, 0.2, 0.8),
                    ),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0,
                        dynamic_friction=1.0,
                        restitution=0.0,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(5.0, 6.0, 0.5),
                ),
            ),
            "goal": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Goal",
                spawn=sim_utils.SphereCfg(
                    radius=goal_radius,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(                        
                        kinematic_enabled=True,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=False,
                    ),                    
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.8, 0.0),
                    ),                    
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=goal_position,
                ),
            ),
        }
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -20.0
    flat_orientation_reward_scale = -5.0
    progress_reward_scale = 450.0
    goal_bonus_reward_scale = 500.0
    speed_penalty_scale = -1.0    
    min_speed_penalty_scale = -10.0
    heading_reward_scale = 0.5
