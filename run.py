"""Launch Isaac Sim Simulator first."""
import argparse

from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
parser.add_argument("--robot_asset_path", type=str, default="data/robot_description/a1arm_description/usd/a1arm/a1arm.usd", help="robot asset path")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""
import gymnasium as gym

import torch

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.controllers import DifferentialIKControllerCfg, DifferentialIKController

from gen2sim.envs.open_cabinet.open_cabinet_env import OpenCabinetEnv, OpenCabinetEnvCfg
from gen2sim.config.assets import A1ARM_CFG, CABINET_CFG

def main():
    robot_cfg = A1ARM_CFG
    robot_cfg.spawn.rigid_props.disable_gravity = False
    robot_cfg.spawn.articulation_props.fix_root_link = True
    cabinet_cfg = CABINET_CFG
    # cabinet_cfg.spawn.articulation_props.fix_root_link = True

    env_cfg = OpenCabinetEnvCfg()
    env_cfg.robot_cfg = robot_cfg
    env_cfg.cabinet_cfg = cabinet_cfg

    env = gym.make("OpenCabinet-v111", cfg=env_cfg)
    env.reset()

    # Simulate physics
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            actions = torch.randn(env.action_space.shape)

            env.step(actions)

if __name__ == "__main__":
    main()