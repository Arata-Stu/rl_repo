from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from src.utils.timers import CudaTimer as Timer
# from src.utils.timers import TimerDummy as Timer

from .base import BaseVAE

class CNN_VAE(BaseVAE):
    def __init__(self, cnn_cfg: DictConfig, latent_dim: int, input_shape: Tuple[int, int, int] = (3, 64, 64)):
        super().__init__(latent_dim, input_shape)
        
        # Encoder
        self.enc_conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

        self.flatten_dim = self._get_flatten_dim()
        self.mu = nn.Linear(self.flatten_dim, latent_dim)
        self.logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim, self.flatten_dim)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec_conv4 = nn.ConvTranspose2d(32, input_shape[0], kernel_size=4, stride=2, padding=1)

        if cnn_cfg.ckpt_path:
            self.load_weights(path=cnn_cfg.ckpt_path)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with Timer(device=x.device, timer_name="VAE: encode"):
            batch_size = x.shape[0]
            out = F.relu(self.enc_conv1(x))
            out = F.relu(self.enc_conv2(out))
            out = F.relu(self.enc_conv3(out))
            out = F.relu(self.enc_conv4(out))
            out = out.view(batch_size, -1)
            
            mu = self.mu(out)
            logvar = self.logvar(out)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        with Timer(device=z.device, timer_name="VAE: decode"):
            batch_size = z.shape[0]
            out = self.dec_fc(z)
            out = out.view(batch_size, 256, self.feature_size, self.feature_size)
            out = F.relu(self.dec_conv1(out))
            out = F.relu(self.dec_conv2(out))
            out = F.relu(self.dec_conv3(out))
            out = torch.sigmoid(self.dec_conv4(out))
        return out

    def _get_flatten_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            out = F.relu(self.enc_conv1(dummy))
            out = F.relu(self.enc_conv2(out))
            out = F.relu(self.enc_conv3(out))
            out = F.relu(self.enc_conv4(out))
            self.feature_size = out.shape[-1]
            return out.numel()
