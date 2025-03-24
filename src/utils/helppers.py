import numpy as np
import torch

def numpy2img_tensor(img: np.ndarray) -> torch.Tensor:
    img = img.astype(np.float32) / 255.0
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float()
    return img

def img_tensor2numpy(img: torch.Tensor) -> np.ndarray:
    img = img.cpu().detach().numpy()
    img = img.transpose((1, 2, 0))
    img = (img * 255).astype(np.uint8)
    return img
