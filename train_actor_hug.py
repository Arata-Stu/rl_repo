import os
import traceback

import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
import hydra
import pygame  # keyboardの代わりにpygameを利用

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
        self.env = get_env(env_cfg=config.envs)
        
        # エージェント、バッファ、VAEの初期化
        state_z_dim = config.vae.latent_dim
        state_vec_dim = 4
        action_dim = 3
        self.agent = get_agents(agent_cfg=config.agent,
                                state_z_dim=state_z_dim,
                                state_vec_dim=state_vec_dim,
                                action_dim=action_dim)
        self.buffer = HugReplayBuffer(max_size=int(config.buffer.size),
                                      state_vec_dim=state_vec_dim,
                                      state_z_dim=state_z_dim,
                                      action_dim=action_dim)
        self.vae = get_vae(vae_cfg=config.vae).to(self.device).eval()

        # TensorBoardの初期化
        self.writer = SummaryWriter(log_dir=config.logger.tensorboard.log_dir)

        self.max_episodes = config.max_episodes
        self.max_steps = config.max_steps
        self.log_interval = config.get("reconstructed_log_interval", 50)
        self.batch_size = config.batch_size
        self.save_ckpt_dir = config.save_ckpt_dir

        # 評価フェーズ用の設定
        self.eval_interval = config.get("eval_interval", 10)
        self.num_eval_episodes = config.get("num_eval_episodes", 1)
        self.record_video = config.get("record_video", True)

        # 人間介入用の変数（初期値）
        self.human_steering = 0.0
        self.human_accel = 0.0
        self.human_brake = 0.0

        # pygame の初期化（小さなウィンドウを作成）
        pygame.init()
        self.pygame_screen = pygame.display.set_mode((100, 100))
        pygame.display.set_caption("Human Intervention Input")

    def update_human_steering(self, current_value, key_pressed, dt=1.0):
        """
        キー入力に基づいて、ステアリング値を更新する。
        応答性を上げるため、インクリメントを大きくし、キーが離されたらすぐに戻るようにする。
        """
        max_steering = 1.0
        increment = 0.05  # 押下時の増分（より敏感）
        decay = 0.05      # キーが離されたときの減衰（すぐに0へ戻す）
        
        if key_pressed == "left":
            current_value = max(-max_steering, current_value - increment)
        elif key_pressed == "right":
            current_value = min(max_steering, current_value + increment)
        else:
            # キーが離れている場合は、速やかに0に戻す
            current_value = 0.0
        return current_value

    def update_human_value(self, current_value, key_pressed, dt=1.0, increment=0.1, decay=0.05, max_value=1.0):
        """
        アクセルやブレーキなど、その他の値の更新。
        キーが押されているときは大きく増加し、離されたら速やかに0に戻す。
        """
        if key_pressed:
            current_value = min(max_value, current_value + increment)
        else:
            current_value = 0.0
        return current_value

    def train(self):
        episode_rewards = []
        top_models = []
        
        try:
            for episode in range(self.max_episodes):
                obs, vehicle_info = self.env.reset()
                episode_reward = 0

                print(f"Episode {episode} started.")
                for step in range(self.max_steps):
                    # pygameイベント処理（キー入力更新のため）
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            self.env.close()
                            return

                    # 画像エンコーディング
                    obs_img = obs["image"].copy()
                    obs_vec = obs["vehicle"]
                    with Timer("Env: Encoding"):
                        obs_img = numpy2img_tensor(obs_img).unsqueeze(0).to(self.device)
                        state = self.vae.obs_to_z(obs_img)
                    with Timer("Env: Decoding"):
                        reconstructed_img = self.vae.decode(state)
                    if step % self.log_interval == 0:
                        reconstructed_img = reconstructed_img.squeeze(0).float().cpu().detach()
                        global_step = episode * self.max_steps + step
                        self.writer.add_image("Actor/Reconstructed/Image", reconstructed_img, global_step)

                    # エージェントが推論したアクション（agent_action）
                    with Timer("Agent Action"):
                        state_vec = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
                        agent_action = self.agent.select_action(state_z=state, state_vec=state_vec, evaluate=False)
                    if not isinstance(agent_action, torch.Tensor):
                        agent_action = torch.tensor(agent_action, dtype=torch.float32, device=self.device)

                    # ---------------------------
                    # キー入力のチェック
                    keys = pygame.key.get_pressed()
                    # 左右キーでステアリング
                    steer_key = None
                    if keys[pygame.K_LEFT]:
                        steer_key = "left"
                    elif keys[pygame.K_RIGHT]:
                        steer_key = "right"
                    # Upキーでアクセル、Downキーでブレーキ
                    accel_pressed = keys[pygame.K_UP]
                    brake_pressed = keys[pygame.K_DOWN]
                    
                    # 介入判定：いずれかのキーが押されていれば介入中とする
                    if steer_key is not None or accel_pressed or brake_pressed:
                        intervention = 1.0
                        # 各値を更新
                        self.human_steering = self.update_human_steering(self.human_steering, steer_key)
                        self.human_accel = self.update_human_value(self.human_accel, accel_pressed)
                        self.human_brake = self.update_human_value(self.human_brake, brake_pressed)
                        # human_action は agent_action をコピーして上書き
                        human_action = agent_action.clone()
                        if human_action.ndim == 1:
                            human_action[0] = self.human_steering   # ステア
                            human_action[1] = self.human_accel       # アクセル
                            human_action[2] = self.human_brake       # ブレーキ
                        else:
                            human_action[0, 0] = self.human_steering
                            human_action[0, 1] = self.human_accel
                            human_action[0, 2] = self.human_brake
                    else:
                        intervention = 0.0
                        # キーが押されていなければ、エージェントのアクションをそのまま使う
                        human_action = agent_action.clone()
                    # ---------------------------
                    
                    # ログ用に両方のアクションを記録
                    self.writer.add_histogram("Actor/AgentAction/Distribution", agent_action, episode)
                    self.writer.add_histogram("Actor/HumanAction/Distribution", human_action, episode)

                    # Gym環境は numpy 配列を要求
                    action_for_env = human_action.cpu().detach().numpy()
                    with Timer("Environment Step"):
                        next_obs, reward, terminated, truncated, info = self.env.step(action_for_env)

                    next_obs_img = next_obs["image"].copy()
                    next_obs_vec = next_obs["vehicle"]
                    with Timer("Next Encoding"):
                        next_obs_img = numpy2img_tensor(next_obs_img).unsqueeze(0).to(self.device)
                        next_state = self.vae.obs_to_z(next_obs_img)

                    with Timer("Buffer Add"):
                        done = terminated or truncated
                        next_state_vec = torch.tensor(next_obs_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
                        # agent_action と human_action の両方を記録
                        self.buffer.add(state_vec, state, agent_action, human_action,
                                        intervention, reward, next_state_vec, next_state, done)

                    episode_reward += reward

                    if len(self.buffer) >= self.batch_size:
                        update_info = self.agent.update(self.buffer, batch_size=self.batch_size)
                        global_step = episode * self.max_steps + step
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

                # トップモデルの保存（上位3件）
                os.makedirs(self.save_ckpt_dir, exist_ok=True)
                if len(top_models) < 3:
                    top_models.append((episode, episode_reward))
                    model_path = f"{self.save_ckpt_dir}/best_{episode_reward:.2f}_ep_{episode}.pt"
                    self.agent.save(model_path, episode)
                else:
                    min_model = min(top_models, key=lambda x: x[1])
                    min_reward = min_model[1]
                    if episode_reward > min_reward:
                        old_model_path = f"{self.save_ckpt_dir}/best_{min_model[1]:.2f}_ep_{min_model[0]}.pt"
                        if os.path.exists(old_model_path):
                            os.remove(old_model_path)
                        top_models.remove(min_model)
                        top_models.append((episode, episode_reward))
                        new_model_path = f"{self.save_ckpt_dir}/best_{episode_reward:.2f}_ep_{episode}.pt"
                        self.agent.save(new_model_path, episode)
                print(f"Episode {episode}: Reward = {episode_reward:.2f}")

                if self.eval_interval > 0 and (episode + 1) % self.eval_interval == 0:
                    self.evaluate(episode)

        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
        finally:
            self.writer.close()
            pygame.quit()
            print("Cleaned up resources.")

    def evaluate(self, current_episode):
        """
        評価フェーズ: エージェントの評価を行い、各エピソードの報酬と
        (設定により)走行動画をTensorBoardに記録します。
        """
        eval_rewards = []
        video_frames = []
        print(f"--- Evaluation Phase at Episode {current_episode} ---")
        
        for ep in range(self.num_eval_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            frames = []

            for step in range(self.max_steps):
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
                    if not isinstance(action, torch.Tensor):
                        action = torch.tensor(action, dtype=torch.float32, device=self.device)
                action_for_env = action.cpu().detach().numpy()
                with Timer("Eval: Environment Step"):
                    next_obs, reward, terminated, truncated, info = self.env.step(action_for_env)
                episode_reward += reward
                obs = next_obs

                if self.record_video and ep == 0:
                    next_frame_tensor = numpy2img_tensor(next_obs["image"].copy())
                    frames.append(next_frame_tensor)

                if terminated or truncated:
                    break

            eval_rewards.append(episode_reward)
            print(f"Evaluation Episode {ep}: Reward = {episode_reward:.2f}")

        avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
        self.writer.add_scalar("Actor/Evaluation/Reward", avg_eval_reward, current_episode)
        print(f"Average Evaluation Reward at Episode {current_episode}: {avg_eval_reward:.2f}")

        if self.record_video and len(frames) > 0:
            video_tensor = torch.stack(frames, dim=0)  # [T, C, H, W]
            video_tensor = video_tensor.unsqueeze(0)     # [B, T, C, H, W]
            self.writer.add_video("Actor/Evaluation/Run", video_tensor, current_episode, fps=60)

@hydra.main(config_path='config', config_name='train_actor', version_base='1.2')
def main(config: DictConfig):
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
