from torchvision.datasets import CocoDetection

class CocoImageDataset(CocoDetection):
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return img