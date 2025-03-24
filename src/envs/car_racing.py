import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class CarRacingWithInfoWrapper(gym.Wrapper):
    def __init__(self, env, width=128, height=128):
        super().__init__(env)
        self.width = width
        self.height = height

        # 画像 + 車両情報 (throttle, brake, steering, velocity)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8),
            "vehicle": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)  # (throttle, brake, steering, velocity)
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_observation(obs), reward, terminated, truncated, info

    def _process_observation(self, obs):
        """ 画像をリサイズし、車両情報をまとめる """
        resized_image = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        car = self.env.unwrapped.car

        # 速度の大きさ (スカラー値)
        velocity = np.linalg.norm(car.hull.linearVelocity)

        # 各ホイールの状態を取得
        throttle = (car.wheels[2].gas + car.wheels[3].gas) / 2  # 後輪のガス平均
        brake = (car.wheels[0].brake + car.wheels[1].brake + car.wheels[2].brake + car.wheels[3].brake) / 4  # 全輪のブレーキ平均
        steering = (car.wheels[0].joint.angle + car.wheels[1].joint.angle) / 2  # 前輪の舵角平均

        # 車両情報を1つの配列にまとめる
        vehicle_state = np.array([throttle, brake, steering, velocity / 100.0], dtype=np.float32)

        return {
            "image": resized_image,
            "vehicle": vehicle_state
        }
