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

from gen2sim.envs.reach import ReachHolonomicMobileEnv, ReachHolonomicMobileEnvCfg
from gen2sim.config.assets import A1ARM_RGM3_CFG

def main():
    robot_cfg = A1ARM_RGM3_CFG

    env_cfg = ReachHolonomicMobileEnvCfg()
    env_cfg.robot_cfg = robot_cfg

    env = gym.make("ReachHolonomicMobile-v1", cfg=env_cfg)
    env.reset()

    # Simulate physics
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            actions = torch.randn(env.action_space.shape)
            actions[:, :3] = 1.

            env.step(actions)

if __name__ == "__main__":
    main()