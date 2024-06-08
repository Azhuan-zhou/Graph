import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, GATv2Conv
from torch_sparse import SparseTensor


def time_sparse(adj_t, timestamps, timestamp):
    device = timestamps.device
    timestamp = torch.tensor(timestamp).to(device)
    temp = torch.isin(timestamps, timestamp)
    edge_time = torch.nonzero(temp).squeeze()
    size = adj_t.sizes()
    row = adj_t.storage.row()
    col = adj_t.storage.col()
    val = adj_t.storage.value()
    mask = torch.isin(val, edge_time)
    # print(mask.sum())
    new_row = row[mask]
    new_col = col[mask]
    new_val = val[mask]
    new_adj_t = SparseTensor(row=new_row, col=new_col, value=new_val, sparse_sizes=size)
    return new_adj_t


class GraphSage(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, batchnorm=True):
        super(GraphSage, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr="mean"))
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        print("batchnorm is,", self.batchnorm)
        self.num_layers = num_layers
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for i in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr="mean"))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr="mean"))
        self.dropout = dropout
        print("dropout is,", self.dropout)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, adjs, time, timestamp, interval):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            # print(i)
            edge_index_time = time_sparse(edge_index, timestamp, [time + t for t in range(0, interval)])
            x = self.convs[i]((x, x_target), edge_index_time)

            if i != self.num_layers - 1:
                if self.batchnorm:
                    x = self.bns[i](x)
                x = F.leaky_relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x, edge_index_time


class GAT(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , layer_heads=[]
                 , batchnorm=True):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.num_layers = num_layers

        if len(layer_heads) > 1:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=layer_heads[0], concat=True))
            if self.batchnorm:
                self.bns = torch.nn.ModuleList()
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels * layer_heads[0]))
            for i in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_channels * layer_heads[i - 1], hidden_channels, heads=layer_heads[i], concat=True))
                if self.batchnorm:
                    self.bns.append(torch.nn.BatchNorm1d(hidden_channels * layer_heads[i - 1]))
            self.convs.append(GATConv(hidden_channels * layer_heads[num_layers - 2]
                                      , out_channels
                                      , heads=layer_heads[num_layers - 1]
                                      , concat=False))
        else:
            self.convs.append(GATConv(in_channels, out_channels, heads=layer_heads[0], concat=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, adjs, time, timestamp, interval):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            edge_index_time = time_sparse(edge_index, timestamp, [time + t for t in range(0, interval)])
            x = self.convs[i]((x, x_target), edge_index_time)
            if i != self.num_layers - 1:
                if self.batchnorm:
                    x = self.bns[i](x)
                x = F.leaky_relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x


class GATv2(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , layer_heads=[]
                 , batchnorm=True):
        super(GATv2, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.num_layers = num_layers

        if len(layer_heads) > 1:
            self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=layer_heads[0], concat=True))
            if self.batchnorm:
                self.bns = torch.nn.ModuleList()
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels * layer_heads[0]))
            for i in range(num_layers - 2):
                self.convs.append(
                    GATv2Conv(hidden_channels * layer_heads[i - 1], hidden_channels, heads=layer_heads[i], concat=True))
                if self.batchnorm:
                    self.bns.append(torch.nn.BatchNorm1d(hidden_channels * layer_heads[i - 1]))
            self.convs.append(GATv2Conv(hidden_channels * layer_heads[num_layers - 2]
                                        , out_channels
                                        , heads=layer_heads[num_layers - 1]
                                        , concat=False))
        else:
            self.convs.append(GATv2Conv(in_channels, out_channels, heads=layer_heads[0], concat=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, adjs, time, timestamp, interval):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            edge_index_time = time_sparse(edge_index, timestamp, [time + t for t in range(0, interval)])
            x = self.convs[i]((x, x_target), edge_index_time)
            if i != self.num_layers - 1:
                if self.batchnorm:
                    x = self.bns[i](x)
                x = F.leaky_relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x
