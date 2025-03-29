from omegaconf import DictConfig
from .td3HUG import TD3HUG
from .sacHUG import SACHUG

def get_agents(agent_cfg: DictConfig, state_z_dim: int, state_vec_dim: int, action_dim: int):
    
    if agent_cfg.name == "TD3-HUG":
        return TD3HUG(state_z_dim=state_z_dim,
                        state_vec_dim=state_vec_dim,
                        action_dim=action_dim,
                        actor_lr=agent_cfg.actor_lr,
                        critic_lr=agent_cfg.critic_lr,
                        gamma=agent_cfg.gamma,
                        tau=agent_cfg.tau,
                        hidden_dim=agent_cfg.hidden_dim,
                        policy_noise=agent_cfg.policy_noise,
                        noise_clip=agent_cfg.noise_clip,
                        policy_delay=agent_cfg.policy_delay,
                        intervention_weight=agent_cfg.intervention_weight,
                        ckpt_path=agent_cfg.ckpt_path)
    elif agent_cfg.name == "SAC-HUG":
        return SACHUG(state_z_dim=state_z_dim,
                      state_vec_dim=state_vec_dim,
                      action_dim=action_dim,
                      actor_lr=agent_cfg.actor_lr,
                      critic_lr=agent_cfg.critic_lr,
                      alpha_lr=agent_cfg.alpha_lr,
                      gamma=agent_cfg.gamma,
                      tau=agent_cfg.tau,
                      hidden_dim=agent_cfg.hidden_dim,
                      intervention_weight=agent_cfg.intervention_weight,
                      ckpt_path=agent_cfg.ckpt_path)
    else:
        raise NotImplementedError(f"Unknown agent: {agent_cfg.name}")