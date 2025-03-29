import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from src.models.actor.actor import ActorSAC   # SAC用の確率的ポリシー（sampleメソッド実装済み）
from src.models.critic.critic import Critic
from .base import BaseAgent

class SACHUG(BaseAgent):
    def __init__(self, 
                 state_z_dim: int,
                 state_vec_dim: int,
                 action_dim: int,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 hidden_dim: int = 256,
                 intervention_weight: float = 1.0,
                 alpha: float = 0.2,
                 ckpt_path: str = None):
        """
        SACエージェント（人間介入を考慮）
        
        Args:
            state_z_dim: 潜在空間の次元数
            state_vec_dim: 元の状態ベクトルの次元数
            action_dim: 行動の次元数
            intervention_weight: 人間介入損失の重み
            alpha: エントロピー正則化の係数
            その他はSACのハイパーパラメータ
        """
        super().__init__(state_z_dim, state_vec_dim, action_dim, gamma, tau, actor_lr, critic_lr)
        
        # Actorは確率的ポリシーを出力するネットワーク（sampleメソッドでサンプルアクション、対数確率、平均行動を返す）
        self.actor = ActorSAC(self.state_dim, action_dim, hidden_dim).to(self.device)
        # Criticは2つのQネットワークを持つ
        self.critic = Critic(self.state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.intervention_weight = intervention_weight  # 介入損失の重み
        self.alpha = alpha  # エントロピー正則化の係数
        
        self.total_iterations = 0

        if ckpt_path:
            self.load(ckpt_path)

    def select_action(self, state_z: torch.Tensor, state_vec: torch.Tensor, evaluate: bool = False):
        """
        アクション選択（評価時は決定論的な行動を返す）
        """
        state = torch.cat([state_vec, state_z], dim=-1).to(self.device)
        with torch.no_grad():
            if evaluate:
                # 決定論的に平均を返す（Actor内で定義されたget_deterministic_actionを利用）
                action = self.actor.get_deterministic_action(state)
            else:
                # sampleメソッドでサンプルされたアクション、対数確率、平均行動を取得
                action, _, _ = self.actor.sample(state)
        return action.cpu().numpy()[0]

    def update(self, buffer, batch_size: int = 64, current_epoch: int = 0):
        """
        SACの学習ステップ
        
        - Criticはターゲットネットワークを用いたTD誤差で更新
        - Actorはエントロピー正則化付きの損失と、介入サンプルに対する教師あり損失を組み合わせて更新
        - 介入サンプルに対しては、エージェントの行動が人間の行動に近づくように学習する
        
        バッチは以下の項目を含む：
        (state_vec, state_z, action, human_action, intervention, reward, next_state_vec, next_state_z, not_done)
        
        current_epoch: 学習時の現在のエポック数（フェードアウト項に利用）
        """
        self.total_iterations += 1

        # リプレイバッファからサンプリング
        sample = buffer.sample(batch_size)
        state_vec = torch.FloatTensor(sample[0]).to(self.device)
        state_z = torch.FloatTensor(sample[1]).to(self.device)
        action = torch.FloatTensor(sample[2]).to(self.device)
        human_action = torch.FloatTensor(sample[3]).to(self.device)
        intervention = torch.FloatTensor(sample[4]).to(self.device)  # 介入フラグ：0または1
        reward = torch.FloatTensor(sample[5]).to(self.device)
        next_state_vec = torch.FloatTensor(sample[6]).to(self.device)
        next_state_z = torch.FloatTensor(sample[7]).to(self.device)
        not_done = torch.FloatTensor(sample[8]).to(self.device)

        # 状態は元の状態ベクトルと潜在表現を結合
        state = torch.cat([state_vec, state_z], dim=-1)
        next_state = torch.cat([next_state_vec, next_state_z], dim=-1)
        
        # --- Criticの更新 ---
        with torch.no_grad():
            # 次状態でのアクションサンプルとその対数確率を取得
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.gamma * (target_q - self.alpha * next_log_prob)
        
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # --- Actorの更新 ---
        # 現在の状態でのサンプルされたアクション、対数確率、そして決定論的行動（平均）を取得
        action_sample, log_prob, mean_action = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, action_sample)
        q_pi = torch.min(q1_pi, q2_pi)
        
        # 非介入サンプルに対する標準的なSACの目的
        # 期待値 E[α log π(a|s) - Q(s,a)]
        loss_policy = self.alpha * log_prob - q_pi
        # 介入していないサンプル（intervention=0）のマスク
        mask_no = 1.0 - intervention
        if mask_no.sum() > 0:
            loss_policy = (loss_policy * mask_no).sum() / mask_no.sum()
        else:
            loss_policy = torch.tensor(0.0).to(self.device)
        
        # 介入サンプルに対する教師あり損失：エージェントの行動（平均）が人間の行動に近づくように
        if intervention.sum() > 0:
            loss_supervised = F.mse_loss(mean_action * intervention, human_action * intervention)
        else:
            loss_supervised = torch.tensor(0.0).to(self.device)
        
        # 時間とともに介入の影響を減衰させるフェードアウト項（例: 0.997^epoch）
        lam = 0.997 ** current_epoch
        
        # 総合的なActor損失
        actor_loss = loss_policy + self.intervention_weight * lam * loss_supervised
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ターゲットネットワークのソフト更新
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item()
        }
