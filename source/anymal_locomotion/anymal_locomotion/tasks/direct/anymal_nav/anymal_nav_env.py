# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse

from .anymal_nav_env_cfg import AnymalNavEnvCfg
from .utils import sixd_to_quat, quat_to_sixd


class AnymalNavEnv(DirectRLEnv):
    cfg: AnymalNavEnvCfg

    def __init__(self, cfg: AnymalNavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._goal_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_goal_dist = torch.zeros(self.num_envs, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "progress_reward",
                "goal_bonus",                       
                "min_speed_penalty",
                "max_speed_penalty",
                "heading_reward",
            ]
        }
        # Get specific body indices
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies("base|.*THIGH")

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor        
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # Obstacles
        if self.cfg.add_obstacles:       
            self._obstacles = self.cfg.obstacles.class_type(self.cfg.obstacles)
            self.scene.rigid_object_collections["obstacles"] = self._obstacles
        else:
            self._obstacles = None
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._target_joint_angles = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._target_joint_angles)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        root_pos = self._robot.data.root_pos_w[:, :2] - self._terrain.env_origins[:, :2]
        goal_vec_w = self._goal_pos - root_pos
        goal_dist = torch.norm(goal_vec_w, dim=-1, keepdim=True)

        goal_vec_w_3d = torch.cat(
            [goal_vec_w, torch.zeros(self.num_envs, 1, device=self.device)],
            dim=-1,
        )

        goal_vec_b = quat_apply_inverse(
            self._robot.data.root_quat_w,
            goal_vec_w_3d,
        )[:, :2]

        obs = torch.cat([
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            goal_vec_b, # Vector to goal in the robot's base frame
            goal_dist, # Distance to the goal
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            self._robot.data.joint_vel,
            self._actions,
        ], dim=-1)

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Reward progress towards the goal
        root_pos = self._robot.data.root_pos_w[:, :2] - self._terrain.env_origins[:, :2]
        goal_vec_w = self._goal_pos - root_pos
        goal_dist = torch.norm(goal_vec_w, dim=-1)
        progress_reward = self._prev_goal_dist - goal_dist
        progress_reward = torch.clamp(progress_reward, max=0.02)        
        self._prev_goal_dist = goal_dist.detach()

        # Heading reward: encourage robot's forward direction to point toward the goal
        goal_vec_w_3d = torch.cat(
            [goal_vec_w, torch.zeros(self.num_envs, 1, device=self.device)],
            dim=-1,
        )

        goal_vec_b = quat_apply_inverse(
            self._robot.data.root_quat_w,
            goal_vec_w_3d,
        )[:, :2]

        heading_error = torch.atan2(goal_vec_b[:, 1], goal_vec_b[:, 0])
        heading_reward = torch.exp(-torch.square(heading_error) / 0.25)

        # Bonus for reaching the goal
        goal_bonus = (goal_dist < self.cfg.goal_radius).float()
        
        speed_xy = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=-1)
        # Penalty for walking too fast
        max_speed_penalty = torch.square(torch.clamp(speed_xy - 0.8, min=0.0))

        # Penalty for standing still
        min_speed_penalty = torch.square(torch.clamp(0.3 - speed_xy, min=0.0))        

        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1)
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        rewards = {            
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale,# * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "progress_reward": progress_reward * self.cfg.progress_reward_scale * self.step_dt,
            "goal_bonus": goal_bonus * self.cfg.goal_bonus_reward_scale * self.step_dt,
            "max_speed_penalty": max_speed_penalty * self.cfg.speed_penalty_scale * self.step_dt,
            "min_speed_penalty": min_speed_penalty * self.cfg.min_speed_penalty_scale * self.step_dt,
            "heading_reward": heading_reward * self.cfg.heading_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0, dim=1)

        root_pos = self._robot.data.root_pos_w[:, :2] - self._terrain.env_origins[:, :2]
        goal_dist = torch.norm(self._goal_pos - root_pos, dim=-1)
        goal_reached = goal_dist < self.cfg.goal_radius        
        died = died | goal_reached
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs and self.num_envs > 1:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        self._goal_pos[env_ids] = torch.tensor(
            [self.cfg.goal_position[0], self.cfg.goal_position[1]], 
            dtype=torch.float32,
            device=self.device
        )
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()

        # Initial y-position of the robot is uniformly distributed in [0, 10]
        default_root_state[:, 1].uniform_(0.0, 10.0)
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        root_pos = self._robot.data.root_pos_w[env_ids, :2] - self._terrain.env_origins[env_ids, :2]
        self._prev_goal_dist[env_ids] = torch.norm(self._goal_pos[env_ids] - root_pos, dim=-1)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
