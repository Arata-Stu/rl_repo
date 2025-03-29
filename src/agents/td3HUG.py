import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from src.models.actor.actor import ActorTD3
from src.models.critic.critic import Critic
from .base import BaseAgent

class TD3HUG(BaseAgent):
    def __init__(self, 
                 state_z_dim: int,
                 state_vec_dim: int,
                 action_dim: int,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005, 
                 hidden_dim: int = 256,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_delay: int = 2,
                 intervention_weight: float = 1.0,
                 ckpt_path: str = None):
        """
        TD3エージェント（人間介入を考慮）
        
        Args:
            state_z_dim: 潜在空間の次元数
            state_vec_dim: 元の状態ベクトルの次元数
            action_dim: 行動の次元数
            intervention_weight: 人間介入の損失項の重み
            その他はTD3のハイパーパラメータ
        """
        super().__init__(state_z_dim, state_vec_dim, action_dim, gamma, tau, actor_lr, critic_lr)
        
        self.actor = ActorTD3(self.state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = ActorTD3(self.state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(self.state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, action_dim, hidden_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_iterations = 0
        self.intervention_weight = intervention_weight  # 人間介入損失の重み
        
        if ckpt_path:
            self.load(ckpt_path)

    def select_action(self, state_z: torch.Tensor, state_vec: torch.Tensor, evaluate: bool = False):
        """
        アクション選択（探索時はノイズを付加）
        """
        state = torch.cat([state_vec, state_z], dim=-1).to(self.device)
        with torch.no_grad():
            action = self.actor(state)
        action = action.cpu().numpy()[0]
        if not evaluate:
            action += np.random.normal(0, self.policy_noise, size=action.shape)
            action = np.clip(action, -1, 1)
        return action

    def update(self, buffer, batch_size: int = 64, current_epoch: int = 0):
        """
        TD3の学習ステップ
        
        - Criticは毎回更新
        - Actorは policy_delay 回に1回更新し、バッチ内で介入データと非介入データを区別して損失を計算
        - ターゲットネットワークのソフト更新を実施
        
        バッチは以下の項目を含むものとする：
        (state_vec, state_z, action, human_action, intervention, reward, next_state_vec, next_state_z, not_done)
        
        current_epoch: 学習時の現在のエポック数（フェードアウト項に利用）
        """
        self.total_iterations += 1

        # リプレイバッファからサンプリング
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


        # 状態は元の状態ベクトルと潜在表現を結合
        state = torch.cat([state_vec, state_z], dim=-1)
        next_state = torch.cat([next_state_vec, next_state_z], dim=-1)
        
        # ターゲットアクションの計算（ノイズ付き）
        with torch.no_grad():
            noise = torch.clamp(
                torch.randn_like(action) * self.policy_noise,
                -self.noise_clip,
                self.noise_clip
            )
            next_action = torch.clamp(self.actor_target(next_state) + noise, -1, 1)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # 現在のQ値の計算とCriticの損失
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = torch.tensor(0.0).to(self.device)
        
        # Actor更新は policy_delay 回に1回行う
        if self.total_iterations % self.policy_delay == 0:
            action_pred = self.actor(state)
            
            # 非介入サンプル（intervention=0）のマスク
            mask_no = 1.0 - intervention  # 形状は (batch, 1) なので自動的にブロードキャスト
            
            # 通常のTD3の目的：非介入サンプルで Q値を最大化
            q1_pred, _ = self.critic(state, action_pred)
            if mask_no.sum() > 0:
                loss_td3 = -(q1_pred * mask_no).sum() / mask_no.sum()
            else:
                loss_td3 = torch.tensor(0.0).to(self.device)
            
            # 介入サンプルについては、エージェントの行動が人間の介入行動に近づくように
            if intervention.sum() > 0:
                loss_supervised = F.mse_loss(action_pred * intervention, human_action * intervention)
            else:
                loss_supervised = torch.tensor(0.0).to(self.device)
            
            # 時間とともに介入の影響を減衰させるフェードアウト項（例: 0.997^epoch）
            lam = 0.997 ** current_epoch
            
            # 総合的なActor損失
            actor_loss = loss_td3 + self.intervention_weight * lam * loss_supervised

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # ターゲットネットワークのソフト更新
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item()
        }
