"""Launch Isaac Sim Simulator first."""
import argparse

from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
parser.add_argument("--config_file_path", type=str, default="data/robot_description/a1arm_description/a1arm.yml", help="config file path")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.terrains import TerrainImporterCfg

# cuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

from gen2sim.config.assets import A1ARM_CFG

def design_scene() -> InteractiveScene:
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    scene = InteractiveScene(scene_cfg)

    _robot_cfg = A1ARM_CFG.copy()
    _robot_cfg.spawn.rigid_props.disable_gravity = False
    _robot_cfg.spawn.articulation_props.fix_root_link = True
    _robot = Articulation(_robot_cfg)
    scene.articulations["robot"] = _robot

    _terrain_cfg = TerrainImporterCfg(
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
    _terrain_cfg.num_envs = scene.cfg.num_envs
    _terrain_cfg.env_spacing = scene.cfg.env_spacing
    _terrain = _terrain_cfg.class_type(_terrain_cfg)

    # clone, filter, and replicate
    scene.clone_environments(copy_from_source=False)
    scene.filter_collisions(global_prim_paths=[_terrain_cfg.prim_path])

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    return scene

def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    scene = design_scene()

    sim.reset()

    decimation = 2
    action_scale = 0.5
    robot_dof_targets = torch.zeros((scene.num_envs, 8), device=scene.device)
    robot_dof_lower_limits = scene.articulations["robot"].data.default_joint_limits[..., 0].to(device=scene.device)
    robot_dof_upper_limits = scene.articulations["robot"].data.default_joint_limits[..., 1].to(device=scene.device)
    robot_ee_link_idx = scene.articulations["robot"].find_bodies("arm_seg6")[0][0]

    tensor_args = TensorDeviceType()
    config_file = load_yaml(args_cli.config_file_path)
    # urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
    # base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
    # ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
    # robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)
    robot_cfg = RobotConfig.from_dict(config_file["robot_cfg"])

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)

    q_sample = ik_solver.sample_configs(scene.num_envs)
    kin_state = ik_solver.fk(q_sample)
    goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)
    result = ik_solver.solve_batch(goal)
    q_solution = result.solution[result.success]

    print(q_sample, q_solution)

    while simulation_app.is_running():
        with torch.inference_mode():
            targets = torch.concat([q_solution, torch.zeros((scene.num_envs, 2), device=scene.device)], dim=-1)
            robot_dof_targets = targets

            for _ in range(decimation):
                scene.articulations["robot"].set_joint_position_target(robot_dof_targets)

                scene.write_data_to_sim()
                sim.step(render=False)

                sim.render()
                scene.update(dt=sim.get_physics_dt())
            
            ee_link_pos = scene.articulations["robot"].data.body_pos_w[:, robot_ee_link_idx]
            ee_link_quat = scene.articulations["robot"].data.body_quat_w[:, robot_ee_link_idx]
            print(goal, Pose(ee_link_pos, ee_link_quat))

if __name__ == "__main__":
    main()