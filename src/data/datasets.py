import os
from torch.utils.data import DataLoader
from torchvision import transforms
from omegaconf import DictConfig

from src.data.img_dataset import ConcatFrameDataset
from src.data.coco_dataset import CocoImageDataset


def get_dataloader(dataset_cfg: DictConfig, mode: str = "train") -> DataLoader:
    """データローダーを取得する"""

    # 前処理の定義
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((dataset_cfg.height, dataset_cfg.width)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((dataset_cfg.height, dataset_cfg.width)),
        transforms.ToTensor(),
    ])

    if mode == "train":
        transform = train_transform
    elif mode == "val":  
        transform = val_transform
    else:
        NotImplementedError(f"mode {mode} is not implemented")

    if dataset_cfg.name == 'img':
        # ToPILImage を transform の先頭に追加
        transform = transforms.Compose([
            transforms.ToPILImage(),  # ← ここが重要！
            *transform.transforms
        ])

        if mode == "train":
            data_root = os.path.join(dataset_cfg.root, "train")
        elif mode == "val":
            data_root = os.path.join(dataset_cfg.root, "val")
        else:
            data_root = dataset_cfg.root
        
        dataset = ConcatFrameDataset(
            root=data_root,
            transform=transform,
        )
    elif dataset_cfg.name == 'coco':
        # COCOデータセットの場合
        if mode == "train":
            data_root = os.path.join(dataset_cfg.root, "images", "train2017")
            ann_file = os.path.join(dataset_cfg.root, "annotations", "instances_train2017.json")
        elif mode == "val":
            data_root = os.path.join(dataset_cfg.root, "images", "val2017")
            ann_file = os.path.join(dataset_cfg.root, "annotations", "instances_val2017.json")
        else:
            data_root = dataset_cfg.root
            ann_file = os.path.join(dataset_cfg.root, "annotations", "instances_test2017.json")
        
        dataset = CocoImageDataset(
            root=data_root,
            annFile=ann_file,
            transform=transform,
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_cfg.name} is not implemented")

    # DataLoaderの作成
    dataloader = DataLoader(
        dataset,
        batch_size=dataset_cfg.batch_size,
        shuffle=(mode == "train"),
        num_workers=dataset_cfg.num_workers,
        pin_memory=True,
    )
    
    return dataloader