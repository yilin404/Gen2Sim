"""
Open-Cabinet environment.
"""

import gymnasium as gym

from . import agents
from .open_cabinet_env import OpenCabinetEnv, OpenCabinetEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="OpenCabinet-v1",
    entry_point=OpenCabinetEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OpenCabinetEnvCfg,
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenCabinetPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)