import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg, Articulation, RigidObjectCfg, RigidObject
from omni.isaac.lab.utils.noise import NoiseModelCfg
from omni.isaac.lab.utils import configclass

from dataclasses import MISSING
from collections.abc import Sequence

import numpy as np
import torch
import gymnasium as gym

import warp as wp

from .base_env import BaseRLEnv, BaseRLEnvCfg


"""
Env and EnvCfg
"""

@configclass
class PickAndPlaceEnvCfg(BaseRLEnvCfg):
    # env
    decimation: int = 2
    episode_length_s: float = 5.0
    seed: int | None = 114514
    observation_noise_model: NoiseModelCfg | None = None
    action_noise_model: NoiseModelCfg | None = None

    single_observation_space: gym.spaces.Box = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(2, 7,)
    )
    single_action_space: gym.spaces.Box = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,))

    # simulation
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)

    # asset
    robot_cfg: ArticulationCfg = MISSING
    to_pick_cfg: RigidObjectCfg = MISSING
    # to_pick_cfg: ArticulationCfg = MISSING
    # to_place_cfg: ArticulationCfg = MISSING
    

class PickAndPlaceEnv(BaseRLEnv):
    cfg: PickAndPlaceEnvCfg

    def __init__(self, cfg: PickAndPlaceEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self) -> None:
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self._to_pick = RigidObject(self.cfg.to_pick_cfg)
        self._robot = Articulation(self.cfg.robot_cfg)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["to_pick"] = self._to_pick
        self.scene.articulations["robot"] = self._robot

    def _configure_gym_env_spaces(self) -> None:
        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.cfg.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.cfg.single_action_space, self.num_envs)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # actions = torch.clamp(actions, min=-1., max=1.)
        # actions = torch.concat([actions, actions[:, [-1]]], dim=-1) # [num_envs, num_joints,]

        # self.actions = actions * (self._robot.data.default_joint_limits[..., 1] - self._robot.data.default_joint_limits[..., 0]) * 0.5 + 0.5 * (self._robot.data.default_joint_limits[..., 1] + self._robot.data.default_joint_limits[..., 0])

        self.actions = torch.clamp(actions, 
                                   min=self._robot.data.default_joint_limits[..., 0],
                                   max=self._robot.data.default_joint_limits[..., 1])
    
    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.actions)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.reset_terminated, self.reset_time_outs

    def _get_rewards(self) -> torch.Tensor:
        pass

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

    def _get_observations(self) -> dict:
        root_pos = self._robot.data.root_pos_w - self.scene.env_origins # [num_envs, 3]
        root_quat = self._robot.data.root_quat_w # [num_envs, 4]

        ee_link_idx = self._robot.find_bodies("arm_segee")[0][0]
        ee_pos = self._robot.data.body_pos_w[:, ee_link_idx] - self.scene.env_origins # [num_envs, 3]
        ee_quat = self._robot.data.body_quat_w[:, ee_link_idx] # [num_envs, 4], wxyz format

        to_pick_pos = self._to_pick.data.root_pos_w - self.scene.env_origins # [num_envs, 3]
        to_pick_quat = self._to_pick.data.root_quat_w # [num_envs, 4], wxyz format

        self._robot.root_physx_view.get_jacobians()

        return dict(policy=torch.concat([to_pick_pos, to_pick_quat], dim=-1),
                    critic=dict(root_pose=torch.concat([root_pos, root_quat], dim=-1),
                                ee_pose=torch.concat([ee_pos, ee_quat], dim=-1),
                                ee_jacobian=self._robot.root_physx_view.get_jacobians()[:, ee_link_idx-1],
                                joint_pos=self._robot.data.joint_pos))


"""
Solution
"""

class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)

class PickAndPlaceSmState:
    """States for the pick and place state machine."""

    APPROACH_ABOVE_OBJECT = wp.constant(0)
    APPROACH_OBJECT = wp.constant(1)
    GRASP_OBJECT = wp.constant(2)
    LIFT_OBJECT = wp.constant(3)


class PickAndPlaceSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.3)

@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=wp.float32),                       # dtype=float32, [num_envs,]
    sm_state: wp.array(dtype=wp.int32),                   # dtype=int32, [num_envs,]
    sm_wait_time: wp.array(dtype=wp.float32),             # dtype=float32, [num_envs,]
    ee_pose: wp.array(dtype=wp.transform),                # dtype=wp.transform, [num_envs, 7]
    gripper_state: wp.array(dtype=wp.float32),            # dtype=float32, [num_envs,]
    object_pose: wp.array(dtype=wp.transform),            # dtype=wp.transform, [num_envs, 7]
    object_approach_offset: wp.array(dtype=wp.transform), # dtype=wp.transform, [num_envs, 7]
    des_object_pose: wp.array(dtype=wp.transform),        # dtype=wp.transform, [num_envs, 7]
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickAndPlaceSmState.APPROACH_ABOVE_OBJECT:
        ee_pose[tid] = wp.transform_multiply(object_approach_offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickAndPlaceSmWaitTime.APPROACH_ABOVE_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickAndPlaceSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickAndPlaceSmState.APPROACH_OBJECT:
        ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickAndPlaceSmWaitTime.APPROACH_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickAndPlaceSmState.GRASP_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickAndPlaceSmState.GRASP_OBJECT:
        ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickAndPlaceSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickAndPlaceSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickAndPlaceSmState.LIFT_OBJECT:
        ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]

class PickAndPlaceSolution:
    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu"):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device

        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device) # [num_envs,]
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device) # [num_envs,]
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device) # [num_envs,]
        self.ee_pose = torch.zeros((self.num_envs, 7), device=self.device) # [num_envs, 7]
        self.gripper_state = torch.full((self.num_envs,), 0.0, device=self.device) # [num_envs,]
        self.object_approach_offset = torch.zeros((self.num_envs, 7), device=self.device) # [num_envs, 7]
        self.object_approach_offset[:, 2] = 0.075
        self.object_approach_offset[:, -1] = 1. # warp expects quaternion as (x, y, z, w)
        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.ee_pose_wp = wp.from_torch(self.ee_pose, wp.transform)
        self.gripper_state_wp = wp.from_torch(self.gripper_state, wp.float32)
        self.object_approach_offset_wp = wp.from_torch(self.object_approach_offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int]):
        """Reset the state machine."""
        # reset state machine
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0
    
    def compute(self, 
                object_pose: torch.Tensor,     # [num_envs, 7]
                des_object_pose: torch.Tensor, # [num_envs, 7]
                ):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        # convert to warp
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                self.ee_pose_wp,
                self.gripper_state_wp,
                object_pose_wp,
                self.object_approach_offset_wp,
                des_object_pose_wp
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        self.ee_pose = self.ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]

        return self.ee_pose, self.gripper_state.unsqueeze(-1)