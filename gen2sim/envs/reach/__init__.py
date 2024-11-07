"""
Open-Cabinet environment.
"""

import gymnasium as gym

# from . import agents
from .reach_holonomic_mobile import ReachHolonomicMobileEnv, ReachHolonomicMobileEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="ReachHolonomicMobile-v1",
    entry_point=ReachHolonomicMobileEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ReachHolonomicMobileEnvCfg,
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenCabinetPPORunnerCfg",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)