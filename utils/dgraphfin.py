from typing import Optional, Callable, List
import os.path as osp
import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from utils.featureProcess import data_process, edge_type_feat


def read_dgraphfin(folder,if_process=False):
    print("read_dgraphfin")
    names = ["dgraphfin.npz"]
    items = [np.load(folder + "/" + name) for name in names]
    x = items[0]["x"]
    y = items[0]["y"].reshape(-1, 1)

    edge_index = items[0]["edge_index"]
    edge_type = items[0]["edge_type"]
    edge_timestamp = items[0]["edge_timestamp"]
    train_mask = items[0]["train_mask"]
    valid_mask = items[0]["valid_mask"]
    test_mask = items[0]["test_mask"]
    # 计算每个结点的出度和入读
    if if_process:
        pass
        # 将x增加一个维度的属性，结点出边类型的出现的次数（21-31），edge_type有11个类别
        # x = np.concatenate((x, edge_type_feat(edge_type, edge_index, x)), axis=1)
    x = torch.tensor(x, dtype=torch.float).contiguous()
    y = torch.tensor(y, dtype=torch.long)
    edge_index = torch.tensor(edge_index.transpose(), dtype=torch.long).contiguous()
    edge_timestamp = torch.tensor(edge_timestamp, dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    train_mask = torch.tensor(train_mask, dtype=torch.long)
    valid_mask = torch.tensor(valid_mask, dtype=torch.long)
    test_mask = torch.tensor(test_mask, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_type, y=y)
    data.train_mask = train_mask
    data.test_mask = test_mask
    data.valid_mask = valid_mask
    data.edge_timestamp = edge_timestamp
    if if_process:
        data = data_process(data)
    return data


class DGraphFin(InMemoryDataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"xygraphp1"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ""

    def __init__(
            self,
            root: str,
            name: str,
            if_process: bool,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
    ):
        self.name = name
        self.nlabels = 2
        self.if_process =if_process
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        names = ["dgraphfin.npz"]
        return names

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        pass

    def process(self):
        data = read_dgraphfin(self.raw_dir, if_process=self.if_process)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.name}()"
