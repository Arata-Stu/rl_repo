import torch
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self,
                 state_z: int,
                 state_vec: int,
                 action_dim: int,
                 gamma: float=0.99,
                 tau: float=0.005,
                 actor_lr: float=3e-4,
                 critic_lr: float=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.state_dim = state_z + state_vec
        self.action_dim = action_dim

        self.actor = None  
        self.critic = None  
        self.actor_optimizer = None  
        self.critic_optimizer = None  

    @abstractmethod
    def select_action(self, state_z: torch.Tensor, state_vec: torch.Tensor, evaluate: bool=False):
        """ 環境との相互作用時にアクションを選択するメソッド（サブクラスで実装） """
        pass

    @abstractmethod
    def update(self, buffer, batch_size: int=64):
        """ バッファからサンプルを取得し、ネットワークの更新を行う（サブクラスで実装） """
        pass

    def save(self, filepath: str, episode: int=None):
        """ モデルのチェックポイントを保存 """
        checkpoint = {
            "episode": episode,
            "actor_state_dict": self.actor.state_dict() if self.actor else None,
            "critic_state_dict": self.critic.state_dict() if self.critic else None,
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict() if self.actor_optimizer else None,
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict() if self.critic_optimizer else None,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load(self, filepath: str):
        """ 保存されたチェックポイントをロード """
        checkpoint = torch.load(filepath, map_location=self.device)
        if self.actor and "actor_state_dict" in checkpoint:
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
        if self.critic and "critic_state_dict" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
        if self.actor_optimizer and "actor_optimizer_state_dict" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        if self.critic_optimizer and "critic_optimizer_state_dict" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        print(f"Checkpoint loaded from {filepath}")
