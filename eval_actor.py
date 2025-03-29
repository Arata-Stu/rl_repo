import os
import torch
from omegaconf import DictConfig, OmegaConf
import hydra

from src.envs.envs import get_env
from src.agents.agents import get_agents
from src.models.vae.vae import get_vae
from src.utils.helppers import numpy2img_tensor

def evaluate_agent(config: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 環境の初期化
    env = get_env(env_cfg=config.envs)
    
    # エージェント、バッファ、VAEの初期化
    state_z_dim = config.vae.latent_dim
    state_vec_dim = 4
    action_dim = 3
    agent = get_agents(agent_cfg=config.agent, state_z_dim=state_z_dim, state_vec_dim=state_vec_dim, action_dim=action_dim)
    vae = get_vae(vae_cfg=config.vae).to(device).eval()

    # 評価ループ
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # 観測画像をテンソルに変換し、VAEで状態表現に変換
        obs_img = obs["image"].copy()
        obs_vec = obs["vehicle"]
        obs_img_tensor = numpy2img_tensor(obs_img).unsqueeze(0).to(device)
        state = vae.obs_to_z(obs_img_tensor)

        # 評価モードでアクション選択
        state_vec = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0).to(device)
        action = agent.select_action(state_z=state, state_vec=state_vec, evaluate=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"評価終了。総報酬: {total_reward:.2f}")

@hydra.main(config_path="config", config_name="eval_actor", version_base="1.2")
def main(config: DictConfig):
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print("------ 設定内容 ------")
    print(OmegaConf.to_yaml(config))
    print("----------------------")
    evaluate_agent(config)

if __name__ == '__main__':
    main()