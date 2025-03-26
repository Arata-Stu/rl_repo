import os
import glob
import h5py
import hdf5plugin
from torch.utils.data import Dataset

class FrameDataset(Dataset):
    """
    1つのh5ファイルから各フレーム（画像）をサンプルとして読み込むデータセットクラスです。
    h5ファイルは 'frames' という名前のデータセットに画像が保存されていることを前提としています。
    transformが指定されている場合は、各フレームに対して前処理を実施します。
    """
    def __init__(self, h5_path, transform=None):
        """
        Args:
            h5_path (str): エピソードのh5ファイルのパス
            transform (callable, optional): 各フレームに適用する前処理（例：リサイズ、正規化など）
        """
        self.h5_path = h5_path
        self.transform = transform
        self.h5_file = None

        # 初回にファイルを開いて全フレーム数を取得
        with h5py.File(self.h5_path, 'r') as f:
            self.num_frames = f['frames'].shape[0]

    def __len__(self):
        return self.num_frames

    def _open_file(self):
        # DataLoaderで複数ワーカーを使用する場合に備え、__getitem__で遅延的にファイルをオープン
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_frames:
            raise IndexError("Index out of range")
        self._open_file()
        frame = self.h5_file['frames'][idx]
        if self.transform:
            frame = self.transform(frame)
        return frame

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()


class ConcatFrameDataset(Dataset):
    """
    rootディレクトリ内の全ての.h5ファイルを探索し、
    それらを連結して1つの大きなフレームデータセットとして扱うクラスです。
    
    各h5ファイルは 'frames' という名前のデータセットに画像が保存されていることを前提としています。
    transformが指定されている場合は、各フレームに対して前処理を実施します。
    """
    def __init__(self, root, transform=None, recursive=False):
        """
        Args:
            root (str): .h5ファイルを探索するルートディレクトリのパス
            transform (callable, optional): 各フレームに適用する前処理
            recursive (bool, optional): Trueの場合、サブディレクトリも再帰的に探索します（デフォルトはFalse）
        """
        self.root = root
        self.transform = transform
        
        # rootディレクトリ内の.h5ファイルのパスを探索
        pattern = os.path.join(root, "**", "*.h5") if recursive else os.path.join(root, "*.h5")
        self.h5_paths = sorted(glob.glob(pattern, recursive=recursive))
        
        # 各h5ファイルからFrameDatasetを作成
        self.datasets = [FrameDataset(path, transform) for path in self.h5_paths]
        
        # 各データセットのサンプル数の累積和を計算（グローバルなインデックス対応用）
        self.cum_lengths = []
        cum = 0
        for ds in self.datasets:
            cum += len(ds)
            self.cum_lengths.append(cum)

    def __len__(self):
        if not self.cum_lengths:
            return 0
        return self.cum_lengths[-1]

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        # グローバルインデックスから対象のデータセットを特定
        dataset_idx = 0
        while idx >= self.cum_lengths[dataset_idx]:
            dataset_idx += 1
        # 対象のデータセット内でのローカルインデックスを計算
        local_idx = idx if dataset_idx == 0 else idx - self.cum_lengths[dataset_idx - 1]
        return self.datasets[dataset_idx][local_idx]
