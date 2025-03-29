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

class ActorSAC(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int=256,
                 log_std_min: int=-20,
                 log_std_max: int=2):
        super(ActorSAC, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.mean = nn.Linear(hidden_dim // 2, action_dim)
        self.log_std = nn.Linear(hidden_dim // 2, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # 重みの初期化
        self.apply(self._initialize_weights)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        with Timer(device=state.device, timer_name="Actor.sample"):
            mean, log_std = self.forward(state)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  
            y_t = torch.tanh(x_t)
            action = y_t
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            mean_action = torch.tanh(mean)
        return action, log_prob, mean_action

    def _initialize_weights(self, module):
        """ 重みとバイアスの適切な初期化 """
        if isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)