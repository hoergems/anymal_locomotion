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

from .anymal_command_tracking_env_cfg import AnymalCommandTrackingEnvCfg

class AnymalCommandTrackingEnv(DirectRLEnv):
    cfg: AnymalCommandTrackingEnvCfg

    def __init__(self, cfg: AnymalCommandTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Low-level joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)

        # Low-level joint position command from the previous step
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # High-level X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "r_gait",
                "undesired_contacts",
                "flat_orientation_l2",
            ]
        }

        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
        #self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies("base|.*THIGH")

    def _setup_scene(self):
        # Add the robot to the scene
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Add a contact sensor
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Create terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Replicate the scene across all parallel environments
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # Store the raw low-level actions from the policy.
        # These are later used for reward computation (e.g. action-rate penalty).
        self._actions = actions.clone()

        # Convert the policy's low-level actions into target joint positions.
        #
        # The low level actions are first scaled and then added to the robot's
        # default standing joint configuration.
        # Consequently, the policy controls joint position offsets rather than
        # absolute joint positions, which is often easier to learn.
        self._target_joint_angles = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        # Send the target joint positions (computed in _pre_physics_step) 
        # to the robot's PD controller.

        # The robot's PD controller then computes the required joint
        # torques to track these target positions during the next physics step.
        self._robot.set_joint_position_target(self._target_joint_angles)

    def _get_observations(self) -> dict:
        # Store the previous actions for reward computation
        self._previous_actions = self._actions.clone()

        # Construct the observation tensor provided to the policy.
        # Each row in the tensor corresponds to one environment.
        obs = torch.cat(
            [
                # Linear velocity (vx, vy, vz) in the robot's base frame.
                self._robot.data.root_lin_vel_b,

                # Angular velocity (roll, pitch, yaw rates) in the robot's base frame.
                self._robot.data.root_ang_vel_b,

                # Gravity vector expressed in the robot's base frame.
                # This provides information about the robot's orientation.
                self._robot.data.projected_gravity_b,

                # Joint angle offsets from the default standing configuration.
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,

                # Joint velocities.
                self._robot.data.joint_vel,

                # Previously applied low-level actions.
                # Including the previous action allows the policy to minimize the
                # action-rate penalty, resulting in smoother control commands.
                self._actions,

                # High-level command (desired forward/lateral velocity and yaw rate).
                # The policy learns to track this command.
                self._commands,
            ], dim=-1
        )

        # Isaac Lab expects observations to be returned as a dictionary.
        # The "policy" entry is passed to the RL policy.        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        #---------------------------------
        # Commanded velocity tracking reward terms
        #---------------------------------
        # linear velocity tracking reward term
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_tracking = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking reward term
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_tracking = torch.exp(-yaw_rate_error / 0.25)

        #---------------------------------
        # Motion regularization terms
        #---------------------------------
        # Penalty term for large z-linear velocities
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # Penalty term for large xy-angular velocities
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # Penalty term for large joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # Penalty term for large joint accelerations
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # Penalty term for rapid changes between consecutive actions (results in smoother control)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # Penalty term for deviations from a flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        #---------------------------------
        # Contact-based reward terms
        #---------------------------------
        # Compute r_gait
        # Check which feet established contact within the last timestep.
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]

        # Time (in seconds) each foot spent in the air before its last contact.
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]

        # Reward or penalize only feet that just made contact.
        # Feet with sufficiently long (> 0.5 s) swing durations receive a positive reward,
        # while shorter swing durations receive a negative reward.
        feet_air_time_reward = (last_air_time - 0.5) * first_contact

        # Sum the reward over all feet.
        r_gait = torch.sum(feet_air_time_reward, dim=1)

        # Only reward stepping if the robot is commanded to move.
        is_moving = torch.norm(self._commands[:, :2], dim=1) > 0.1
        r_gait *= is_moving

        # Compute penalty term for undesired contacts.
        # Contact forces (over the history buffer) acting on the undesired body parts.
        net_contact_forces = self._contact_sensor.data.net_forces_w_history[:, :, self._undesired_contact_body_ids]

        # Check whether each undesired body part is in contact with the terrain.
        is_contact = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > 1.0

        # Count the number of undesired contacts.
        undesired_contacts = torch.sum(is_contact, dim=1)  

        # Combine the individual reward terms with their corresponding weights.
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_tracking * self.cfg.lin_vel_reward_scale,
            "track_ang_vel_z_exp": yaw_rate_tracking * self.cfg.yaw_rate_reward_scale,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale,
            "r_gait": r_gait * self.cfg.feet_air_time_reward_scale,
            "undesired_contacts": undesired_contacts * self.cfg.undesired_contact_reward_scale,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale,
        }

        # Sum up all reward terms
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Scale reward by the step duration
        reward *= self.step_dt

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value * self.step_dt
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # An episode can terminate because:
        # 1. The maximum episode length has been reached.
        # 2. The robot's base body collides with the gound.

        # Check if we reached the maximum episode length
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # History of net contact force vectors for every body in the robot
        # Using a short history makes contact detection more robust than relying
        # on a single physics timestep.
        net_contact_forces = self._contact_sensor.data.net_forces_w_history

        # Extract the contact forces acting on the robot's base body.
        base_net_contact_forces = net_contact_forces[:, :, self._base_id]
        # Compute the magnitude of each contact force vector.
        norm_base_net_contact_forces = torch.norm(base_net_contact_forces, dim=-1)
        # Maximum contact force magnitude over the contact sensor's history.
        max_contact_forces = torch.max(norm_base_net_contact_forces, dim=1)[0]
        # Episode terminates if the maximum contact force magnitude exceeds the threshold
        died = torch.any(max_contact_forces > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        self.randomize_target_velocity(env_ids)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
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

    def randomize_target_velocity(self, env_ids: torch.Tensor | None = None):
        """Randomize the target velocity commands for the specified environments.

        Args:
            env_ids: Environment indices to randomize. If None, all environments
                are randomized.
        """
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self._commands[env_ids] = torch.empty_like(
            self._commands[env_ids]
        ).uniform_(-1.0, 1.0)
