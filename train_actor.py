import os
import traceback

import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
import hydra

from src.envs.envs import get_env
from src.agents.agents import get_agents
from src.buffers.HUG_replay_buffer import HugReplayBuffer
from src.models.vae.vae import get_vae
from src.utils.helppers import numpy2img_tensor
from src.utils.timers import Timer

class Trainer:
    def __init__(self, config: DictConfig):
        self.config = config
        print('------ Configuration ------')
        print(OmegaConf.to_yaml(config))
        print('---------------------------')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 環境の初期化
        self.env = get_env(env_cfg=config.env)
        

        # エージェント、バッファ、VAEの初期化
        state_z_dim = config.vae.latent_dim
        state_vec_dim = 4
        action_dim = 3
        self.agent = get_agents(agent_cfg=config.agent, state_z_dim=state_z_dim, state_vec_dim=state_vec_dim, action_dim=action_dim)
        self.buffer = HugReplayBuffer(max_size=config.buffer.max_size, state_vec_dim=state_vec_dim, state_z_dim=state_z_dim, action_dim=action_dim)
        self.vae = get_vae(vae_cfg=config.vae).to(self.device).eval()

        # TensorBoardの初期化
        self.writer = SummaryWriter(log_dir=config.logger.tensorboard.log_dir)

        self.max_episodes = config.max_episodes
        self.max_steps = config.max_steps
        self.log_interval = config.get("reconstructed_log_interval", 50)
        self.batch_size = config.batch_size
        self.save_ckpt_dir = config.save_ckpt_dir

        # 評価フェーズ用の設定
        self.eval_interval = config.get("eval_interval", 10)          # 例: 10エピソードごとに評価
        self.num_eval_episodes = config.get("num_eval_episodes", 1)      # 評価時のエピソード数
        self.record_video = config.get("record_video", True)             # 評価時に動画を記録するかどうか

    def train(self):
        episode_rewards = []
        top_models = []
        
        try:
            for episode in range(self.max_episodes):
                obs, vehicle_info = self.env.reset()
                episode_reward = 0

                print(f"Episode {episode} started.")
                for step in range(self.max_steps):
                    obs_img = obs["image"].copy()
                    obs_vec = obs["vehicle"]
                    with Timer("Env: Encoding"):
                        obs_img = numpy2img_tensor(obs_img).unsqueeze(0).to(self.device)
                        state = self.vae.obs_to_z(obs_img)

                    with Timer("Env: Decoding"):
                        reconstucted_img = self.vae.decode(state)
                    if step % self.log_interval == 0:
                        reconstucted_img = reconstucted_img.squeeze(0).float().cpu().detach()
                        global_step = episode * self.max_steps + step
                        self.writer.add_image("Actor/Reconstructed/Image", reconstucted_img, global_step)

                    with Timer("Agent Action"):
                        state_vec = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
                        action = self.agent.select_action(state_z=state, state_vec=state_vec, evaluate=False)
                    self.writer.add_histogram("Actor/Action/Distribution", action, episode)

                    with Timer("Environment Step"):
                        next_obs, reward, terminated, truncated, info = self.env.step(action)

                    next_obs_img = next_obs["image"].copy()
                    next_obs_vec = next_obs["vehicle"]
                    with Timer("Next Encoding"):
                        next_obs_img = numpy2img_tensor(next_obs_img).unsqueeze(0).to(self.device)
                        next_state = self.vae.obs_to_z(next_obs_img)

                    with Timer("Buffer Add"):
                        done = terminated or truncated
                        next_state_vec = torch.tensor(next_obs_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
                        self.buffer.add(state, state_vec, action, reward, next_state, next_state_vec, done)

                    episode_reward += reward

                    if len(self.buffer) >= self.batch_size:
                        update_info = self.agent.update(self.buffer, batch_size=self.batch_size)
                        global_step = episode * self.max_steps + step

                        # 動的に key を処理して TensorBoard に記録
                        for key, value in update_info.items():
                            self.writer.add_scalar(f"Actor/Loss/{key}", value, global_step)


                    obs = next_obs

                    if terminated or truncated:
                        with Timer("Environment Reset"):
                            obs, vehicle_info = self.env.reset()
                            print(f"Episode {episode}: Step {step} terminated (terminated: {terminated} or truncated: {truncated})")
                        break

                episode_rewards.append(episode_reward)
                self.writer.add_scalar("Actor/Reward/Episode", episode_reward, episode)

                # トップモデルの保存処理（上位3件を保存）
                os.makedirs(self.save_ckpt_dir, exist_ok=True)
                if len(top_models) < 3:
                    top_models.append((episode, episode_reward))
                    model_path = f"{self.save_ckpt_dir}/best_{episode_reward:.2f}_ep_{episode}.pt"
                    self.agent.save(model_path, episode)
                else:
                    # 現在のトップモデルの中で最小の報酬を取得
                    min_model = min(top_models, key=lambda x: x[1])
                    min_reward = min_model[1]

                    if episode_reward > min_reward:
                        # 古いモデルのファイルを削除
                        old_model_path = f"{self.save_ckpt_dir}/best_{min_model[1]:.2f}_ep_{min_model[0]}.pt"
                        if os.path.exists(old_model_path):
                            os.remove(old_model_path)

                        # リストから古いモデルを削除
                        top_models.remove(min_model)

                        # 新しいモデルを追加
                        top_models.append((episode, episode_reward))
                        new_model_path = f"{self.save_ckpt_dir}/best_{episode_reward:.2f}_ep_{episode}.pt"
                        self.agent.save(new_model_path, episode)
                print(f"Episode {episode}: Reward = {episode_reward:.2f}")

                # 評価フェーズの実施
                if self.eval_interval > 0 and (episode + 1) % self.eval_interval == 0:
                    self.evaluate(episode)

        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
        finally:
            self.writer.close()
            print("Cleaned up resources.")

    def evaluate(self, current_episode):
        """
        評価フェーズ: エージェントの評価を行い、各エピソードの報酬と
        (設定により)走行動画をTensorBoardに記録します。
        """
        eval_rewards = []
        video_frames = []  # 動画記録用のフレームリスト
        print(f"--- Evaluation Phase at Episode {current_episode} ---")
        
        for ep in range(self.num_eval_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            frames = []  # 各評価エピソードごとに動画フレームを記録

            for step in range(self.max_steps):
                # 動画を記録する場合、最初の評価エピソードのフレームを保存
                if self.record_video and ep == 0:
                    frame_tensor = numpy2img_tensor(obs["image"].copy())
                    frames.append(frame_tensor)
                    
                with Timer("Eval: Encoding"):
                    obs_img = numpy2img_tensor(obs["image"].copy()).unsqueeze(0).to(self.device)
                    state = self.vae.obs_to_z(obs_img)
                with Timer("Eval: Agent Action"):
                    obs_vec = obs["vehicle"]
                    state_vec = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
                    action = self.agent.select_action(state_z=state, state_vec=state_vec, evaluate=True)
                with Timer("Eval: Environment Step"):
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                obs = next_obs

                if self.record_video and ep == 0:
                    next_frame_tensor = numpy2img_tensor(next_obs["image"].copy())
                    frames.append(next_frame_tensor)

                if terminated or truncated:
                    break

            eval_rewards.append(episode_reward)
            print(f"Evaluation Episode {ep}: Reward = {episode_reward:.2f}")

        # 平均評価報酬をTensorBoardに記録
        avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
        self.writer.add_scalar("Actor/Evaluation/Reward", avg_eval_reward, current_episode)
        print(f"Average Evaluation Reward at Episode {current_episode}: {avg_eval_reward:.2f}")

        # 最初の評価エピソードの動画をTensorBoardに記録
        if self.record_video and len(frames) > 0:
            # framesは各フレームが [C, H, W] のテンソルのリスト
            video_tensor = torch.stack(frames, dim=0)  # [T, C, H, W]
            # TensorBoardのadd_videoは形状[B, T, C, H, W]を期待するので1次元追加
            video_tensor = video_tensor.unsqueeze(0)
            self.writer.add_video("Actor/Evaluation/Run", video_tensor, current_episode, fps=60)

@hydra.main(config_path='config', config_name='train', version_base='1.2')
def main(config: DictConfig):
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
