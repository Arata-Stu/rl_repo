import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from src.utils.timers import CudaTimer as Timer
# from src.utils.timers import TimerDummy as Timer

class BaseVAE(nn.Module):
    def __init__(self, latent_dim: int, input_shape: Tuple[int, int, int] = (3, 64, 64)):
        super().__init__()
        self.latent_size = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def latent(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        with Timer(device=mu.device, timer_name="VAE: latent"):
            sigma = torch.exp(0.5 * logvar)
            eps = torch.randn_like(logvar).to(self.device)
            z = mu + eps * sigma
        return z
    
    def obs_to_z(self, x: torch.Tensor) -> torch.Tensor:
        with Timer(device=x.device, timer_name="VAE: obs_to_z"):
            mu, logvar = self.encode(x)
            z = self.latent(mu, logvar)
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with Timer(device=x.device, timer_name="VAE: encode + decode"):
            mu, logvar = self.encode(x)
            z = self.latent(mu, logvar)
            out = self.decode(z)
        return out, mu, logvar
    
    def vae_loss(self, out, y, mu, logvar):
        BCE = F.mse_loss(out, y, reduction="sum")
        KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KL, BCE, KL
    
    def get_latent_size(self):
        return self.latent_size
    
    def load_weights(self, path: str, strict: bool = True):
        """
        指定されたパスから重みをロードする。
        .pth ファイルと .ckpt ファイルの両方に対応。

        Args:
            path (str): 重みファイルのパス
            strict (bool): strict モードでロードするかどうか
        """
        if path.endswith(".pth"):
            # PyTorch 標準の state_dict をロード
            state_dict = torch.load(path, map_location=self.device)
            self.load_state_dict(state_dict, strict=strict)
            print(f"Loaded .pth weights from {path}")
        
        elif path.endswith(".ckpt"):
            # PyTorch Lightning のチェックポイント
            checkpoint = torch.load(path, map_location=self.device)
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                # Lightning の state_dict には "model." というプレフィックスがついていることがある
                new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
                self.load_state_dict(new_state_dict, strict=strict)
                print(f"Loaded .ckpt weights from {path}")
            else:
                raise ValueError(f"Invalid .ckpt file format: {path}")
        
        else:
            raise ValueError(f"Unsupported file format: {path}")

