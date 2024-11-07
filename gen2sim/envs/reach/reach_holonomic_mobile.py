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
class ReachHolonomicMobileEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s: float = 8.3333 # 500 timesteps
    decimation: int = 2
    seed: int | None = 114514
    num_actions: int = 3 + 8
    num_observations: int = (3 + 8) * 2
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

    # control
    robot_mobile_joint_names = ["root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"]
    robot_arm_joint_names = ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6", "gripper1_axis", "gripper2_axis"]

class ReachHolonomicMobileEnv(DirectRLEnv):
    cfg: ReachHolonomicMobileEnvCfg

    def __init__(self, cfg: ReachHolonomicMobileEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        assert len(self.cfg.robot_mobile_joint_names) == 3, ""
        assert len(self.cfg.robot_mobile_joint_names) + len(self.cfg.robot_arm_joint_names) == self._robot.num_joints, ""

        self.num_mobile_dof = len(self.cfg.robot_mobile_joint_names)
        self.num_arm_dof = len(self.cfg.robot_arm_joint_names)
        self.robot_mobile_dof_targets = torch.zeros((self.num_envs, self.num_mobile_dof), device=self.device) # [num_envs, 3]
        self.robot_arm_dof_targets = torch.zeros((self.num_envs, self.num_arm_dof), device=self.device) # [num_envs, num_arm_dof]

        self.robot_mobile_joint_indices = self._robot.find_joints(self.cfg.robot_mobile_joint_names)[0] # 3
        self.robot_arm_joint_indices = self._robot.find_joints(self.cfg.robot_arm_joint_names)[0] # num_arm_dof
        self.robot_ee_link_idx = self._robot.find_bodies("arm_seg6")[0][0]

        self.robot_arm_dof_lower_limits = self._robot.data.default_joint_limits[:, self.robot_arm_joint_indices, 0] # [num_envs, num_arm_dof]
        self.robot_arm_dof_upper_limits = self._robot.data.default_joint_limits[:, self.robot_arm_joint_indices, 1] # [num_envs, num_arm_dof]

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot_cfg)
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
        self.actions = actions.clone()

        self.robot_mobile_dof_targets[:] = self.actions[..., :3] # [num_envs, 3], x轴线速度 + y轴线速度 + 角速度
        arm_targets = torch.clamp(self.actions[..., 3:], 
                                  min=self.robot_arm_dof_lower_limits,
                                  max=self.robot_arm_dof_upper_limits) # [num_envs, num_arm_dof], 机械臂关节位置
        self.robot_arm_dof_targets[:] = arm_targets
        
    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.robot_arm_dof_targets, joint_ids=self.robot_arm_joint_indices)
        self._robot.set_joint_velocity_target(self.robot_mobile_dof_targets, joint_ids=self.robot_mobile_joint_indices)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self.reset_terminated
        truncated = self.reset_time_outs
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        return None

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # robot state
        mobile_joint_pos = torch.zeros((len(env_ids), self.num_mobile_dof), device=self.device)
        mobile_joint_vel = torch.zeros_like(mobile_joint_pos)
        self._robot.set_joint_position_target(mobile_joint_pos,
                                              joint_ids=self.robot_mobile_joint_indices,
                                              env_ids=env_ids)
        self._robot.write_joint_state_to_sim(mobile_joint_pos, mobile_joint_vel,
                                             joint_ids=self.robot_mobile_joint_indices,
                                             env_ids=env_ids)

        arm_joint_pos = sample_uniform(
            self.robot_arm_dof_lower_limits[0],
            self.robot_arm_dof_upper_limits[0],
            (len(env_ids), self.num_arm_dof),
            self.device,
        )
        arm_joint_pos = torch.clamp(arm_joint_pos, self.robot_arm_dof_lower_limits[env_ids], self.robot_arm_dof_upper_limits[env_ids])
        arm_joint_vel = torch.zeros_like(arm_joint_pos)
        self._robot.set_joint_position_target(arm_joint_pos, 
                                              joint_ids=self.robot_arm_joint_indices, 
                                              env_ids=env_ids)
        self._robot.write_joint_state_to_sim(arm_joint_pos, arm_joint_vel, 
                                             joint_ids=self.robot_arm_joint_indices, 
                                             env_ids=env_ids)

    def _get_observations(self) -> dict:
        robot_dof_pos = self._robot.data.joint_pos
        robot_dof_vel = self._robot.data.joint_vel

        return dict(policy=torch.concat([robot_dof_pos, robot_dof_vel], dim=-1))