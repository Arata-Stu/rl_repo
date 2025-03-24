from omegaconf import DictConfig
from .HUG_replay_buffer import HugReplayBuffer

def get_buffers(buffer_cfg: DictConfig, state_z_dim: int, state_vec_dim, action_dim: int):
    if buffer_cfg.name == "off_policy_HUG":
        return HugReplayBuffer(max_size=int(buffer_cfg.size), state_vec_dim=state_vec_dim, state_z_dim=state_z_dim, action_dim=action_dim)
    else:
        raise ValueError(f"Unexpected buffer name: {buffer_cfg.name}")