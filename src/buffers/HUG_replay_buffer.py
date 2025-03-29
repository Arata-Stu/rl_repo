import numpy as np
import torch
from typing import Union

from .base import ReplayBufferBase

class HugReplayBuffer(ReplayBufferBase):
    def __init__(self, max_size, state_vec_dim, state_z_dim, action_dim):
        """
        Args:
            max_size (int): バッファの最大サイズ
            state_vec_dim (int): 元の状態ベクトルの次元数
            state_z_dim (int): 潜在空間（例：VAEのlatent vector）の次元数
            action_dim (int): 行動の次元数
        """
        self.max_size = max_size
        self.ptr = 0         # 現在の書き込み位置
        self.size = 0        # 現在のデータ数

        # 状態情報（元の状態と潜在表現）を保存
        self.state_vec = np.zeros((max_size, state_vec_dim), dtype=np.float32)
        self.state_z = np.zeros((max_size, state_z_dim), dtype=np.float32)
        self.next_state_vec = np.zeros((max_size, state_vec_dim), dtype=np.float32)
        self.next_state_z = np.zeros((max_size, state_z_dim), dtype=np.float32)

        # 行動情報
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        # 人間の介入時の行動（エキスパートの行動）を保存
        self.human_action = np.zeros((max_size, action_dim), dtype=np.float32)
        # 介入フラグ（介入があった場合は1、なければ0）
        self.intervention = np.zeros((max_size, 1), dtype=np.float32)

        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        # doneフラグは1（継続状態）または0（エピソード終了）で管理
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state_vec, state_z, action, human_action, intervention, reward, next_state_vec, next_state_z, done):
        """
        経験をバッファに追加する。

        Args:
            state_vec (np.array): 元の状態ベクトル
            state_z (np.array): 潜在空間の状態（例：VAEのlatent vector）
            action (np.array): エージェントが実行した行動
            human_action (np.array): 人間の介入時の行動（介入がない場合は適当なデフォルト値を設定）
            intervention (float): 介入フラグ（介入があれば1、なければ0）
            reward (float): 受け取った報酬
            next_state_vec (np.array): 次の状態の元の状態ベクトル
            next_state_z (np.array): 次の状態の潜在空間情報
            done (bool): エピソード終了フラグ
        """
        self.state_vec[self.ptr] = self._to_numpy(state_vec)
        self.state_z[self.ptr] = self._to_numpy(state_z)
        self.action[self.ptr] = self._to_numpy(action)
        self.human_action[self.ptr] = self._to_numpy(human_action)
        self.intervention[self.ptr] = self._to_numpy(intervention)
        self.reward[self.ptr] = self._to_numpy(reward)
        self.next_state_vec[self.ptr] = self._to_numpy(next_state_vec)
        self.next_state_z[self.ptr] = self._to_numpy(next_state_z)
        self.not_done[self.ptr] = 1.0 - float(done)
        
        # リングバッファ方式で古いデータを上書き
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        ランダムにミニバッチをサンプリングする。

        Args:
            batch_size (int): サンプリングするバッチサイズ

        Returns:
            dict: 各項目をキーとする辞書形式
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "state_vec": self.state_vec[indices],
            "state_z": self.state_z[indices],
            "action": self.action[indices],
            "human_action": self.human_action[indices],
            "intervention": self.intervention[indices],
            "reward": self.reward[indices],
            "next_state_vec": self.next_state_vec[indices],
            "next_state_z": self.next_state_z[indices],
            "done": 1.0 - self.not_done[indices]  # ←SACに合わせて「done」に変更してもOK
        }

