import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from omegaconf import DictConfig

from src.envs.car_racing import CarRacingWithInfoWrapper

def get_env(env_cfg: DictConfig) -> gym.Env:
    """環境を取得する"""
    env = gym.make(env_cfg.name)
    env = TimeLimit(env, max_episode_steps=env_cfg.num_steps)

    if env_cfg.name == "CarRacing-v3":
        env = CarRacingWithInfoWrapper(env=env, width=env_cfg.width, height=env_cfg.height)
    else:
        NotImplementedError(f"Unsupported environment: {env_cfg.name}")


    return env