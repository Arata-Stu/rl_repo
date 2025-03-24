import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from src.utils.timers import CudaTimer as Timer
# from src.utils.timers import TimerDummy as Timer

class ActorTD3(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int=256):
        super(ActorTD3, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, action_dim)

        # 重みの初期化
        self.apply(self._initialize_weights)

    def forward(self, state):
        with Timer(device=state.device, timer_name="Actor.forward"):
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            action = torch.tanh(self.fc3(x))
        return action

    def _initialize_weights(self, module):
        """ 重みとバイアスの適切な初期化 """
        if isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
