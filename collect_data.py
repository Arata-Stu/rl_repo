import os
import numpy as np
import hydra
import pygame
import h5py  
import hdf5plugin
from omegaconf import DictConfig, OmegaConf

from src.envs.envs import get_env
from src.utils.timers import Timer as Timer
from src.utils.preprocessing import _blosc_opts
@hydra.main(config_path="config", config_name="collect_data", version_base="1.2")
def main(config: DictConfig):
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # 環境のセットアップ
    env_cfg = config.env
    env = get_env(env_cfg)
    
    mode = config.mode  # 'manual' または 'random'

    # pygame の初期化（manual モードのみ）
    if mode == "manual":
        pygame.init()
        screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("CarRacing Control")
        clock = pygame.time.Clock()

    # 画像保存用のベースディレクトリ
    save_dir = config.output_dir
    os.makedirs(save_dir, exist_ok=True)

    max_episodes = config.num_episodes
    max_steps = config.num_steps

    episode_count = 0
    while episode_count < max_episodes:
        obs, info = env.reset()
        step = 0
        episode_reward = 0
        done = False

        # エピソード中の全フレームを保存するリスト
        episode_frames = []

        while not done and step < max_steps:
            if mode == "manual":
                screen.fill((0, 0, 0))
                keys = pygame.key.get_pressed()
                action = np.array([0.0, 0.0, 0.0])

                if keys[pygame.K_LEFT]:
                    action[0] = -0.3
                if keys[pygame.K_RIGHT]:
                    action[0] = 0.3
                if keys[pygame.K_UP]:
                    action[1] = 0.3
                if keys[pygame.K_DOWN]:
                    action[2] = 0.3

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        env.close()
                        return
                clock.tick(30)
            
            else:  # 'random' モード
                action = env.action_space.sample()

            # 環境をステップ実行
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # obsが辞書の場合は画像部分を取得
            if isinstance(obs, dict):
                obs = obs["image"]

            # numpy配列であればリストに追加
            if isinstance(obs, np.ndarray):
                episode_frames.append(obs)

            step += 1

        print(f"Episode {episode_count + 1}/{max_episodes} finished. Total reward: {episode_reward:.2f}")

        # 収集したフレームをnumpy配列に変換
        episode_frames = np.stack(episode_frames, axis=0)
        episode_h5_path = os.path.join(save_dir, f"ep{episode_count:03d}.h5")
        
                # Blosc圧縮を使って保存
        with h5py.File(episode_h5_path, "w") as hf:
            hf.create_dataset("frames", data=episode_frames, **_blosc_opts())

        episode_count += 1

    env.close()
    if mode == "manual":
        pygame.quit()

if __name__ == "__main__":
    main()
