import hydra
from omegaconf import DictConfig, OmegaConf
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision  

from src.models.vae.vae import get_vae
from src.data.datasets import get_dataloader  

class CheckpointManager:
    """
    指定された基本ディレクトリにチェックポイントを保存し、
    topk 個のベストな重みのみを保持する管理クラス。
    """
    def __init__(self, base_dir, topk):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.topk = topk
        self.checkpoints = []  # (loss, filepath) のリスト

    def update(self, epoch, loss, model):
        filename = f"vae_epoch{epoch}.pth"
        filepath = os.path.join(self.base_dir, filename)
        # チェックポイントの保存
        torch.save(model.state_dict(), filepath)
        print(f"Checkpoint saved: {filepath}")
        # チェックポイントリストに追加
        self.checkpoints.append((loss, filepath))
        # 損失の昇順（低いほうが良い）でソート
        self.checkpoints.sort(key=lambda x: x[0])
        # topk を超える場合は、最も損失が高い（リストの末尾）のチェックポイントを削除
        if len(self.checkpoints) > self.topk:
            worst_loss, worst_path = self.checkpoints.pop()
            if os.path.exists(worst_path):
                os.remove(worst_path)
                print(f"Removed old checkpoint: {worst_path}")

@hydra.main(config_path="config", config_name="train_vae", version_base="1.2")
def main(config: DictConfig):
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    latent_dim = config.vae.latent_dim
    input_shape = tuple(config.input_shape)  # [C, H, W]
    lr = config.lr
    num_epochs = config.num_epochs

    # latent_dim と input_shape の情報を含むディレクトリ名を生成
    input_shape_str = "x".join(map(str, input_shape))
    checkpoint_dir = os.path.join(config.checkpoint.base_dir, f"vae_latent_{latent_dim}_shape_{input_shape_str}")
    log_dir = os.path.join(config.tensorboard.log_dir, f"vae_latent_{latent_dim}_shape_{input_shape_str}")

    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"TensorBoard log directory: {log_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## VAEを定義
    vae = get_vae(config.vae).to(device)

    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # 学習データローダーと評価（検証）データローダーを取得
    train_loader = get_dataloader(config.dataset, split="train")
    val_loader = get_dataloader(config.dataset, split="val")

    # CheckpointManager と TensorBoard の初期化
    checkpoint_manager = CheckpointManager(checkpoint_dir, config.checkpoint.topk)
    writer = SummaryWriter(log_dir=log_dir)

    def train_epoch(model, train_loader, optimizer, device, epoch):
        model.train()
        train_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [Train]")
        for batch_idx, data in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss, bce, kl = model.vae_loss(recon, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = train_loss / len(train_loader.dataset)
        print(f"====> Epoch {epoch} Train Average loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate_epoch(model, val_loader, device, epoch, writer):
        model.eval()
        val_loss = 0
        with torch.no_grad():
            pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch} [Val]")
            for batch_idx, data in pbar:
                data = data.to(device)
                recon, mu, logvar = model(data)
                loss, bce, kl = model.vae_loss(recon, data, mu, logvar)
                val_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_val_loss = val_loss / len(val_loader.dataset)
        print(f"====> Epoch {epoch} Validation Average loss: {avg_val_loss:.4f}")

        # TensorBoard に画像を記録するため、サンプルデータを取得
        sample_data = next(iter(val_loader)).to(device)
        recon, _, _ = model(sample_data)
        
        # torchvision の make_grid で元画像と再構成画像それぞれをグリッドに変換
        grid_original = torchvision.utils.make_grid(sample_data, nrow=8, normalize=True)
        grid_recon = torchvision.utils.make_grid(recon, nrow=8, normalize=True)
        
        # 横方向に結合して比較しやすいようにする（dim=2 は幅方向）
        grid_combined = torch.cat([grid_original, grid_recon], dim=2)
        
        writer.add_image("Comparison (Original | Reconstructed)", grid_combined, epoch)

        return avg_val_loss

    for epoch in range(1, num_epochs + 1):
        avg_train_loss = train_epoch(vae, train_loader, optimizer, device, epoch)
        avg_val_loss = evaluate_epoch(vae, val_loader, device, epoch, writer)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        checkpoint_manager.update(epoch, avg_val_loss, vae)

    writer.close()

if __name__ == "__main__":
    main()
