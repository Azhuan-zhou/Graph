import torch
import os
from datetime import datetime
import shutil
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_sparse import SparseTensor
def prepare_folder(name, model_name):
    model_dir = f'./model_results/{name}/{model_name}'

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    return model_dir


def prepare_tune_folder(name, model_name):
    str_time = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    tune_model_dir = f'./tune_results/{name}/{model_name}/{str_time}/'

    if os.path.exists(tune_model_dir):
        print(f'rm tune_model_dir {tune_model_dir}')
        shutil.rmtree(tune_model_dir)
    os.makedirs(tune_model_dir)
    print(f'make tune_model_dir {tune_model_dir}')
    return tune_model_dir


def save_preds_and_params(parameters, preds, model, file):
    save_dict = {'parameters': parameters, 'preds': preds, 'params': model.state_dict()
        , 'nparams': sum(p.numel() for p in model.parameters())}
    torch.save(save_dict, file)
    return


def draw(edge_index):
    graph = Data(edge_index=edge_index)
    graph = to_networkx(graph)
    nx.draw(graph, with_labels=graph.nodes)
    plt.show()


def sparse_to_date(sparse_tensor):
    if not isinstance(sparse_tensor, SparseTensor):
        raise TypeError('your input type is {}; we need SparseTensor'.format(type(sparse_tensor)))
    # 将稀疏张量转换为edge_index形式
    coo_tensor = sparse_tensor.coo()
    value = coo_tensor[2]
    edge_index = torch.stack([coo_tensor[0], coo_tensor[1]], dim=0)
    data = Data(edge_index=edge_index, x=value)
    return data


def sparse_to_edge(sparse_tensor, adj_t=None):
    if not isinstance(sparse_tensor, SparseTensor):
        raise TypeError('your input type is {}; we need SparseTensor'.format(type(sparse_tensor)))
    # 将稀疏张量转换为edge_index形式
    coo_tensor = sparse_tensor.coo()
    edge_attr = coo_tensor[2]
    if adj_t:
        edge_index = torch.stack([coo_tensor[1], coo_tensor[0]], dim=0)
    else:
        edge_index = torch.stack([coo_tensor[0], coo_tensor[1]], dim=0)
    return edge_index, edge_attr


