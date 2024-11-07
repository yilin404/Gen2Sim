import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg, Articulation
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.utils import configclass

from dataclasses import MISSING
from collections.abc import Sequence

import torch

"""
Env and EnvCfg
"""

@configclass
class OpenCabinetEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s: float = 8.3333 # 500 timesteps
    decimation: int = 2
    seed: int | None = 114514
    num_actions: int = 8
    num_observations: int = 8 + 7 + 4 + 4 + 7
    num_states: int = 0

    # simulation
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)

    # asset
    robot_cfg: ArticulationCfg = MISSING
    cabinet_cfg: ArticulationCfg = MISSING
    # ground plane
    terrain_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # reward scales
    dist_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05

    # other scale
    action_scale = 7.5

class OpenCabinetEnv(DirectRLEnv):
    cfg: OpenCabinetEnvCfg

    def __init__(self, cfg: OpenCabinetEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        self.cabinet_root_pos_init = self._cabinet.data.root_pos_w - self.scene.env_origins
        self.cabinet_root_quat_init = self._cabinet.data.root_quat_w

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.default_joint_limits[..., 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.default_joint_limits[..., 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("gripper1_axis")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("gripper2_axis")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.robot_ee_link_idx = self._robot.find_bodies("arm_seg6")[0][0]
        self.robot_left_gripper_link_idx = self._robot.find_bodies("gripper1")[0][0]
        self.robot_right_gripper_link_idx = self._robot.find_bodies("gripper2")[0][0]
        self.drawer_link_idx = self._cabinet.find_bodies("drawer_top")[0][0]

    def _setup_scene(self) -> None:
        self._cabinet = Articulation(self.cfg.cabinet_cfg)
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["cabinet"] = self._cabinet
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain_cfg.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain_cfg.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain_cfg.class_type(self.cfg.terrain_cfg)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain_cfg.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, 
                          actions: torch.Tensor, # [num_envs, num_actions]
                          ) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.step_dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
    
    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._cabinet.data.joint_pos[:, 3] > 0.39
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # distance from robot ee link to the drawer
        robot_ee_pos = self._robot.data.body_pos_w[:, self.robot_ee_link_idx]
        drawer_pos = self._cabinet.data.body_pos_w[:, self.drawer_link_idx]
        d_ee = torch.norm(robot_ee_pos - drawer_pos, p=2, dim=-1)
        dist_reward_ee = 1.0 / (1.0 + d_ee**2)

        # distance from robot left gripper to the drawer
        robot_left_gripper_pos = self._robot.data.body_pos_w[:, self.robot_left_gripper_link_idx]
        drawer_pos = self._cabinet.data.body_pos_w[:, self.drawer_link_idx]
        d_left_gripper = torch.norm(robot_left_gripper_pos - drawer_pos, p=2, dim=-1)
        dist_reward_left_gripper = 1.0 / (1.0 + d_left_gripper**2)

        # distance from robot right gripper to the drawer
        robot_right_gripper_pos = self._robot.data.body_pos_w[:, self.robot_right_gripper_link_idx]
        drawer_pos = self._cabinet.data.body_pos_w[:, self.drawer_link_idx]
        d_right_gripper = torch.norm(robot_right_gripper_pos - drawer_pos, p=2, dim=-1)
        dist_reward_right_gripper = 1.0 / (1.0 + d_right_gripper**2)

        # how far the cabinet has been opened out
        cabinet_dof_pos = self._cabinet.data.joint_pos
        open_reward = cabinet_dof_pos[:, 3]  # drawer_top_joint

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum((self.actions)**2, dim=-1)

        reward = self.cfg.dist_reward_scale * dist_reward_ee + \
                 self.cfg.dist_reward_scale * dist_reward_left_gripper + \
                 self.cfg.dist_reward_scale * dist_reward_right_gripper + \
                 self.cfg.open_reward_scale * open_reward + \
                 -self.cfg.action_penalty_scale * action_penalty

        self.extras["log"] = {
            "dist_reward_ee": (self.cfg.dist_reward_scale * dist_reward_ee).mean(),
            "dist_reward_left_gripper": (self.cfg.dist_reward_scale * dist_reward_left_gripper).mean(),
            "dist_reward_right_gripper": (self.cfg.dist_reward_scale * dist_reward_right_gripper).mean(),
            "open_reward": (self.cfg.open_reward_scale * open_reward).mean(),
            "action_penalty": (self.cfg.action_penalty_scale * action_penalty).mean(),
        }

        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # robot state
        joint_pos = sample_uniform(
            self.robot_dof_lower_limits[0],
            self.robot_dof_upper_limits[0],
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits[env_ids], self.robot_dof_upper_limits[env_ids])
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # cabinet state
        zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)
        # cabinet_root_delta_pos = torch.randn_like(self.cabinet_root_pos_init[env_ids]).clamp(min=-0.35, max=0.35)
        # cabinet_root_delta_pos[..., -1] = 0.
        # cabinet_root_pos = self.cabinet_root_pos_init[env_ids] + cabinet_root_delta_pos
        # cabinet_root_pos += self.scene.env_origins[env_ids]
        # self._cabinet.write_root_pose_to_sim(torch.concat([cabinet_root_pos,
        #                                                    self.cabinet_root_quat_init[env_ids]], dim=-1), env_ids=env_ids)

    def _get_observations(self) -> dict:
        robot_dof_pos = self._robot.data.joint_pos
        robot_root_pos = self._robot.data.root_pos_w - self.scene.env_origins
        root_root_quat = self._robot.data.root_quat_w

        cabinet_dof_pos = self._cabinet.data.joint_pos
        cabinet_dof_vel = self._cabinet.data.joint_vel
        cabinet_root_pos = self._cabinet.data.root_pos_w - self.scene.env_origins
        cabinet_root_quat = self._cabinet.data.root_quat_w

        return dict(policy=torch.concat([robot_dof_pos, robot_root_pos, root_root_quat,
                                         cabinet_dof_pos, cabinet_dof_vel, cabinet_root_pos, cabinet_root_quat], dim=-1))