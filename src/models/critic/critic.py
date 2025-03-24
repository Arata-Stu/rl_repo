import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from src.utils.timers import CudaTimer as Timer
# from src.utils.timers import TimerDummy as Timer

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int=256):
        super(Critic, self).__init__()
        # Q1 ネットワーク
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.q1 = nn.Linear(hidden_dim // 2, 1)
        # Q2 ネットワーク（Double Q Learning 用）
        self.fc3 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.q2 = nn.Linear(hidden_dim // 2, 1)

        # 重みの初期化
        self.apply(self._initialize_weights)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        with Timer(device=state.device, timer_name="Critic.forward"):
            xu = torch.cat([state, action], dim=1)
            x1 = F.relu(self.fc1(xu))
            x1 = F.relu(self.fc2(x1))
            q1 = self.q1(x1)
            x2 = F.relu(self.fc3(xu))
            x2 = F.relu(self.fc4(x2))
            q2 = self.q2(x2)
        return q1, q2

    def _initialize_weights(self, module: nn.Module):
        """ 重みとバイアスの適切な初期化 """
        if isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
