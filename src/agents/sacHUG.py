import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np  

from src.models.actor.actor import ActorSAC
from src.models.critic.critic import Critic
from .base import BaseAgent

class SACAgent(BaseAgent):
    def __init__(self,
                 state_z_dim: int,
                 state_vec_dim: int,
                 action_dim: int,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 alpha_lr: float = 3e-4, 
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 hidden_dim: int = 256,
                 intervention_weight: float = 1.0,
                 ckpt_path: str = None):
        super().__init__(state_z_dim, state_vec_dim, action_dim, gamma, tau, actor_lr, critic_lr)

        self.actor = ActorSAC(self.state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(self.state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -action_dim

        # 人間介入損失の重み
        self.intervention_weight = intervention_weight

        if ckpt_path:
            self.load(ckpt_path)

    def select_action(self, state_z: torch.Tensor, state_vec: torch.Tensor, evaluate: bool = False) -> np.ndarray:
        state = torch.cat([state_z, state_vec], dim=-1)
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)
        return action.cpu().numpy()[0]

    def update(self, buffer, batch_size: int = 64, current_epoch: int = 0):
        # サンプルは辞書形式で以下のキーを含むものとする：
        # "state_z", "state_vec", "action", "human_action", "intervention", "reward",
        # "next_state_z", "next_state_vec", "done"
        sample = buffer.sample(batch_size)
        state_z = torch.FloatTensor(sample["state_z"]).to(self.device)
        state_vec = torch.FloatTensor(sample["state_vec"]).to(self.device)
        action = torch.FloatTensor(sample["action"]).to(self.device)
        reward = torch.FloatTensor(sample["reward"]).to(self.device)
        next_state_z = torch.FloatTensor(sample["next_state_z"]).to(self.device)
        next_state_vec = torch.FloatTensor(sample["next_state_vec"]).to(self.device)
        done = torch.FloatTensor(sample["done"]).to(self.device)
        human_action = torch.FloatTensor(sample["human_action"]).to(self.device)
        intervention = torch.FloatTensor(sample["intervention"]).to(self.device)  # 介入フラグ：0または1

        next_state = torch.cat([next_state_z, next_state_vec], dim=-1)
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - torch.exp(self.log_alpha) * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        state = torch.cat([state_z, state_vec], dim=-1)
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor更新（人間介入の考慮）
        # actor.sampleは (sampled action, log_prob, mean_action) を返す前提
        action_new, log_prob, mean_action = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, action_new)
        q_new = torch.min(q1_new, q2_new)

        # 介入していないサンプル（intervention==0）に対しては従来のSACの目的
        mask_no = 1.0 - intervention  # ブロードキャストを仮定
        if mask_no.sum() > 0:
            loss_policy = (torch.exp(self.log_alpha) * log_prob - q_new)
            loss_policy = (loss_policy * mask_no).sum() / mask_no.sum()
        else:
            loss_policy = 0.0

        # 介入サンプルに対しては、人間の行動に合わせる教師あり損失（MSE）を計算
        if intervention.sum() > 0:
            loss_supervised = F.mse_loss(mean_action * intervention, human_action * intervention)
        else:
            loss_supervised = 0.0

        # 時間とともに介入影響を減衰させるフェードアウト項
        lam = 0.997 ** current_epoch
        actor_loss = loss_policy + self.intervention_weight * lam * loss_supervised

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
        }
