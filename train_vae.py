import hydra
from omegaconf import DictConfig, OmegaConf
import os
import torch
import torch.optim as optim
from tqdm import tqdm
import torchvision
import wandb
from torch.utils.tensorboard import SummaryWriter

from src.models.vae.vae import get_vae
from src.data.datasets import get_dataloader  
import os
import torch
import wandb

class CheckpointManager:
    """
    指定された基本ディレクトリにチェックポイントを保存し、
    topk 個のベストな重みのみを保持する管理クラス。
    wandb を利用している場合は、チェックポイントをアーティファクトとしてアップロードする。
    """
    def __init__(self, base_dir, topk, logger_type):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.topk = topk
        self.checkpoints = []  # (loss, filepath) のリスト
        self.logger_type = logger_type

    def update(self, epoch, loss, model):
        # last.ckpt の保存（常に上書き）
        last_ckpt_path = os.path.join(self.base_dir, "last.ckpt")
        torch.save(model.state_dict(), last_ckpt_path)
        print(f"Last checkpoint saved: {last_ckpt_path}")

        # 評価損失を名前に含めた保存ファイル名
        raw_filename = f"vae_epoch{epoch}_val{loss:.4f}.pth"
        raw_filepath = os.path.join(self.base_dir, raw_filename)
        torch.save(model.state_dict(), raw_filepath)
        print(f"Checkpoint saved: {raw_filepath}")

        # wandb アップロード処理
        if self.logger_type == "wandb":
            artifact = wandb.Artifact(f"model_checkpoint_epoch_{epoch}", type="model")
            artifact.add_file(raw_filepath)
            artifact.add_file(last_ckpt_path)
            wandb.log_artifact(artifact)
            print(f"Checkpoint artifact logged to wandb for epoch {epoch}")

        # topk の更新と整理
        self.checkpoints.append((loss, raw_filepath))
        self.checkpoints.sort(key=lambda x: x[0])  # 小さい順に並び替え
        self._rename_topk_checkpoints()

    def _rename_topk_checkpoints(self):
        # topkのファイルを _top{rank} 付きに整理
        new_checkpoints = []
        for idx, (loss, old_path) in enumerate(self.checkpoints[:self.topk]):
            new_filename = f"vae_epoch*_val{loss:.4f}_top{idx+1}.pth"

            # 元のファイル名から epoch を抽出（正規表現で抽出しても可）
            basename = os.path.basename(old_path)
            parts = basename.split("_")
            epoch_part = next((p for p in parts if p.startswith("epoch")), "epochUnknown")
            new_filename = f"vae_{epoch_part}_val{loss:.4f}_top{idx+1}.pth"
            new_path = os.path.join(self.base_dir, new_filename)

            # リネーム処理
            if old_path != new_path:
                if os.path.exists(new_path):
                    os.remove(new_path)
                os.rename(old_path, new_path)
                print(f"Renamed {old_path} -> {new_path}")
            new_checkpoints.append((loss, new_path))

        # topk外を削除
        for loss, path in self.checkpoints[self.topk:]:
            if os.path.exists(path):
                os.remove(path)
                print(f"Removed old checkpoint: {path}")

        self.checkpoints = new_checkpoints


class LoggerWrapper:
    """
    TensorBoard と wandb の切り替え可能なロガークラス。
    config.logger.type に "tensorboard" または "wandb" を指定してください。
    """
    def __init__(self, config, latent_dim, input_shape_str):
        logger_cfg = config.logger
        self.logger_type = logger_cfg.type.lower()
        if self.logger_type == "tensorboard":
            # 新しい構造に合わせて config.logger.tensorboard.log_dir を参照
            log_dir = os.path.join(logger_cfg.tensorboard.log_dir, f"vae_latent_{latent_dim}_shape_{input_shape_str}")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard log directory: {log_dir}")
        elif self.logger_type == "wandb":
            # wandb 用の project/group は config.logger.wandb 以下に定義
            wandb.init(
                project=logger_cfg.wandb.project,
                group=logger_cfg.wandb.group,
                config=OmegaConf.to_container(config, resolve=True)
            )
            print(f"wandb initialized with project: {logger_cfg.wandb.project}, group: {logger_cfg.wandb.group}")
        else:
            raise ValueError(f"Unsupported logger type: {self.logger_type}")
    
    def add_scalar(self, tag, value, step):
        if self.logger_type == "tensorboard":
            self.writer.add_scalar(tag, value, step)
        elif self.logger_type == "wandb":
            wandb.log({tag: value, "step": step})
    
    def add_image(self, tag, img_tensor, step):
        if self.logger_type == "tensorboard":
            self.writer.add_image(tag, img_tensor, step)
        elif self.logger_type == "wandb":
            # wandb は numpy 配列を受け付けるため、.cpu().numpy() に変換
            wandb.log({tag: wandb.Image(img_tensor.cpu().numpy()), "step": step})
    
    def close(self):
        if self.logger_type == "tensorboard":
            self.writer.close()
        elif self.logger_type == "wandb":
            wandb.finish()

def train_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    train_loss = 0
    with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [Train]") as pbar:
        for batch_idx, data in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss, bce, kl = model.vae_loss(recon, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # 必要に応じて pbar.set_postfix(loss=f"{loss.item():.4f}") を使えます
    avg_loss = train_loss / len(train_loader.dataset)
    print(f"====> Epoch {epoch} Train Average loss: {avg_loss:.4f}")
    return avg_loss

def evaluate_epoch(model, val_loader, device, epoch, logger):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        with tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch} [Val]") as pbar:
            for batch_idx, data in pbar:
                data = data.to(device)
                recon, mu, logvar = model(data)
                loss, bce, kl = model.vae_loss(recon, data, mu, logvar)
                val_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
    avg_val_loss = val_loss / len(val_loader.dataset)
    print(f"====> Epoch {epoch} Validation Average loss: {avg_val_loss:.4f}")

    # サンプルデータを使って画像を取得しログする
    sample_data = next(iter(val_loader)).to(device)
    recon, _, _ = model(sample_data)
    grid_original = torchvision.utils.make_grid(sample_data, nrow=8, normalize=True)
    grid_recon = torchvision.utils.make_grid(recon, nrow=8, normalize=True)
    grid_combined = torch.cat([grid_original, grid_recon], dim=2)
    logger.add_image("Comparison (Original | Reconstructed)", grid_combined, epoch)
    
    return avg_val_loss


@hydra.main(config_path="config", config_name="train_vae", version_base="1.2")
def main(config: DictConfig):
    # コンフィグの解決と表示
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print("------ Configuration ------")
    print(OmegaConf.to_yaml(config))
    print("---------------------------")

    latent_dim = config.vae.latent_dim
    # vae の input_shape を利用
    input_shape = tuple(config.vae.input_shape)
    lr = config.lr
    num_epochs = config.num_epochs

    # latent_dim と input_shape の情報を含むディレクトリ名を生成
    input_shape_str = "x".join(map(str, input_shape))
    checkpoint_dir = os.path.join(
        config.checkpoint.base_dir,
        f"vae_latent_{latent_dim}_shape_{input_shape_str}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # VAE の生成と最適化手法の設定
    vae = get_vae(config.vae).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # 学習・評価用データローダーの取得
    train_loader = get_dataloader(config.dataset, mode="train")
    val_loader = get_dataloader(config.dataset, mode="val")

    # ロガーとチェックポイントマネージャの初期化
    logger = LoggerWrapper(config, latent_dim, input_shape_str)
    checkpoint_manager = CheckpointManager(checkpoint_dir, config.checkpoint.topk, config.logger.type.lower())

    for epoch in range(1, num_epochs + 1):
        avg_train_loss = train_epoch(vae, train_loader, optimizer, device, epoch)
        avg_val_loss = evaluate_epoch(vae, val_loader, device, epoch, logger)
        logger.add_scalar("Loss/train", avg_train_loss, epoch)
        logger.add_scalar("Loss/val", avg_val_loss, epoch)
        checkpoint_manager.update(epoch, avg_val_loss, vae)

    logger.close()

if __name__ == "__main__":
    main()
