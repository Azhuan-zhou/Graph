import torch
import torch.nn.functional as F
import torch_geometric
import torch_scatter as scatter
from torch import Tensor
import numpy as np
from scipy.sparse import csr_matrix


def edge_type_feat(edge_type, edge_index, x):
    edge_type_adj = csr_matrix(
        (edge_type, (edge_index[:, 0], edge_index[:, 1])),
        shape=(x.shape[0], x.shape[0]),
    )
    edge_type_feat = np.zeros((x.shape[0], 11))
    data, indptr = edge_type_adj.data, edge_type_adj.indptr
    for i in range(x.shape[0]):
        row = data[indptr[i]: indptr[i + 1]]  # edge_type_adj中一行的数据
        unique, counts = np.unique(row, return_counts=True)
        for j, k in zip(unique, counts):
            edge_type_feat[i, j - 1] = k  # 边的第j个type,以及出现的次数K
    return edge_type_feat  # 有11列，属性是某种edge_type出现的次数

def add_degree_feature(x: Tensor, edge_index: Tensor):
    row, col = edge_index
    in_degree = torch_geometric.utils.degree(col, x.size(0), x.dtype)

    out_degree = torch_geometric.utils.degree(row, x.size(0), x.dtype)
    return torch.cat([x, in_degree.view(-1, 1), out_degree.view(-1, 1)], dim=1)




def add_feature_flag(x):
    # 缺失值处理， 增加flag，用于判断原始属性中有缺失值的属性
    feature_flag = torch.zeros_like(x[:, :17])
    feature_flag[x[:, :17] == -1] = 1
    # 将所有1替换为0
    x[x == -1] = 0
    return torch.cat((x, feature_flag), dim=1)


def add_label_feature(x, y):
    y = y.clone()
    # All fraudulent nodes are temporarily considered as normal users to simulate the scenario of mining fraudulent users from normal users.
    # 将所有的欺诈者的标签设置为0，模拟假装正常用户的欺诈者，结点的属性（+3）
    y[y == 1] = 0
    y_one_hot = F.one_hot(y).squeeze()
    return torch.cat((x, y_one_hot[:, :-1]), dim=1)


def add_label_counts(x, edge_index, y):
    # 处理背景结点和前景结点
    y = y.clone().squeeze()
    # 背景结点
    background_nodes = torch.logical_or(y == 2, y == 3)
    foreground_nodes = torch.logical_and(y != 2, y != 3)
    # 将背景结点设置为1
    y[background_nodes] = 1
    # 将前景结点设置为0
    y[foreground_nodes] = 0

    row, col = edge_index
    # 起始结点的one_hot编码
    a = F.one_hot(y[col])
    # 终止结点的one_hot编码
    b = F.one_hot(y[row])
    # 结点的前景属性和背景属性
    # 起始结点
    temp = scatter.scatter(  a, row, dim=0, dim_size=y.size(0), reduce="sum")
    # 终止结点
    temp += scatter.scatter(b, col, dim=0, dim_size=y.size(0), reduce="sum")
    #(结点作为前景结点在边中出现次数，结点作为背景结点在边中出现次数)
    return torch.cat([x, temp.to(x)], dim=1)


def cos_sim_sum(x, edge_index):
    # 计算相似系数之和/相似系数平均
    row, col = edge_index
    sim = F.cosine_similarity(x[row], x[col])  # 计算边相连结点的余弦相似度
    sim_sum = scatter.scatter(sim, row, dim=0, dim_size=x.size(0), reduce="mean")
    return torch.cat([x, torch.unsqueeze(sim_sum, dim=1)], dim=1)


def to_undirected(edge_index, edge_attr, edge_timestamp):

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    edge_timestamp = torch.cat([edge_timestamp, edge_timestamp], dim=0)
    return edge_index, edge_attr, edge_timestamp


def data_process(data):
    # 边连接的结点，边的属性，边的时间戳
    edge_index, edge_attr, edge_timestamp = (
        data.edge_index,
        data.edge_attr,
        data.edge_timestamp,
    )
    x = data.x
    # 增加出度和入度(+1)
    #x = add_degree_feature(x, edge_index)
    # 增加结点与其指向结点向量的余弦相似度(+1)
    #x = cos_sim_sum(x, edge_index)
    # 将有向边转化为无向边
    edge_index, edge_attr, edge_timestamp = to_undirected(
        edge_index, edge_attr, edge_timestamp
    )
    # 左边数字大的结点
    mask = edge_index[0] < edge_index[1]
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask]
    edge_timestamp = edge_timestamp[mask]
    data.edge_index, data.edge_attr, data.edge_timestamp = to_undirected(
        edge_index, edge_attr, edge_timestamp
    )
    # 前景结点和背景结点的one-hot，以及出现的的次数（+2），此时
    #x = add_label_counts(x, edge_index, data.y)
    # 仿真假装normal的fraud（+4）
    #x = add_label_feature(x, data.y)
    data.x = x
    if data.y.dim() == 2:
        data.y = data.y.squeeze(1)
    data.edge_attr = data.edge_attr - 1
    return data

