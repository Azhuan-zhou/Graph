import config as cfg
import os.path

from utils.dgraphfin import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from torch_geometric.loader import NeighborSampler
from models.Temporal_Spacial import TS
from logger import Logger
from tqdm import tqdm
import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T

import pandas as pd

eval_metric = 'auc'

param_dict = {
    'lr': cfg.lr,
    'num_layers': cfg.num_layers,
    'hidden_size': cfg.hidden_size,
    'embedding_size': cfg.embedding_size,
    'dropout': cfg.dropout,
    'batchnorm': cfg.batchnorm,
    'l2': cfg.l2,
    'mem_slots': cfg.mem_slots,
    'head_size': cfg.head_size,
    'num_heads': cfg.num_heads,
    'num_blocks': cfg.num_blocks,
    'forget_bias': cfg.forget_bias,
    'input_bias': cfg.input_bias,
    'interval': cfg.interval,
    'gate_style': cfg.gate_style,
    'attention_mlp_layers': cfg.attention_mlp_layers,
    'key_size': cfg.key_size,
    'return_all_outputs': cfg.return_all_outputs,
    'embedding_model': cfg.embedding_model,
    'num_edge_attr': cfg.num_edge_attr
}

if cfg.embedding_model == 'GAT' or cfg.embedding_model == 'GATv2':
    param_dict['layer_heads'] = cfg.layer_heads


def train(epoch, device, train_loader, model, data, train_idx, optimizer):
    model.train()
    pbar = tqdm(total=train_idx.size(0), ncols=80)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        optimizer.zero_grad()
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        timestamp = data.edge_timestamp.to(device)
        memory = model.initial_state(batch_size).to(device)
        x = data.x[n_id].to(device)
        out, _ = model(x, memory, adjs, timestamp,data.edge_attr)
        target = data.y[n_id[:batch_size]].to(device)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        pbar.update(batch_size)
    pbar.close()
    loss = total_loss / len(train_loader)
    return loss


@torch.no_grad()
def test(layer_loader, device, model, data, split_idx, evaluator):
    # data.y is labels of shape (N, )
    model.eval()
    timestamp = data.edge_timestamp.to(device)
    x_all = data.x.to(device)
    outputs = []
    pbar = tqdm(total=x_all.size(0), ncols=80)
    pbar.set_description('Evaluating')
    #——————————————test
    for batch_size, n_id, adjs in layer_loader:
        adjs = [adj.to(device) for adj in adjs]
        memory = model.initial_state(batch_size).to(device)
        out = model(x_all[n_id], memory, adjs, timestamp,data.edge_attr)
        outputs.append(out)
        pbar.update(batch_size)
    outputs = torch.cat(outputs, dim=0)
    y_pred = outputs.exp()  # (N,num_classes)

    losses, eval_results = dict(), dict()
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        node_id = node_id.to(device)
        y = data.y.to(device)
        losses[key] = F.nll_loss(outputs[node_id], y[node_id]).item()
        eval_results[key] = evaluator.eval(y[node_id], y_pred[node_id])[eval_metric]
    pbar.close()
    return eval_results, losses, y_pred


def main():
    parser = argparse.ArgumentParser(description='minibatch_gnn_models')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model', type=str, default='TS')
    parser.add_argument('--dataset', type=str, default='DGraphFin')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--embedding_model', type=str, default='GraphSage')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--save_step', type=int, default=5)
    parser.add_argument('--if_process', type=bool, default=True)
    args = parser.parse_args()
    print(args)

    device = f'{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = DGraphFin(root='./dataset/', name=args.dataset,if_process=args.if_process, transform=T.ToSparseTensor())

    nlabels = dataset.nlabels
    if args.dataset == 'DGraphFin': nlabels = 2
    data = dataset[0]

    print('x dim = ', data.x.size(-1))
    if data.y.ndim == 2:
        data.y = data.y.squeeze(dim=-1)

    split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}
    print(split_idx)
    fold = args.fold
    if split_idx['train'].dim() > 1 and split_idx['train'].shape[1] > 1:
        kfolds = True
        print('There are {} folds of splits'.format(split_idx['train'].shape[1]))
        split_idx['train'] = split_idx['train'][:, fold]
        split_idx['valid'] = split_idx['valid'][:, fold]
        split_idx['test'] = split_idx['test'][:, fold]
    else:
        kfolds = False
    train_idx = split_idx['train']
    # 准备模型训练结果的文件夹
    result_dir = prepare_folder(args.dataset, args.model)
    print('result_dir:', result_dir)
    train_loader = NeighborSampler(data.adj_t, node_idx=train_idx, sizes=[15, 5], batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=8)
    layer_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[15, 5], batch_size=args.batch_size, shuffle=False,
                                   num_workers=8)
    print('Initialize Model......')
    model_params = param_dict.copy()
    model_params.pop('lr')
    model_params.pop('l2')
    model = TS(input_size=data.x.size(-1), output_size=nlabels, **model_params).to(device)
    print("Successfully initialized")
    for name, param in model.named_parameters():
        print(name, param.shape, param.dtype)
    print(model)

    evaluator = Evaluator(eval_metric)
    logger = Logger(args.runs, args)
    for run in range(args.runs):
        import gc
        gc.collect()
        print(sum(p.numel() for p in model.parameters()))

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=param_dict['lr'], weight_decay=param_dict['l2'])

        best_valid = 0
        min_valid_loss = 1e8
        best_out = None

        for epoch in range(1, args.epochs + 1):
            loss = train(epoch, device, train_loader, model, data, train_idx, optimizer)
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, ')

            # -------------evaluating-----------
            if epoch % args.log_steps == 0:
                eval_results, losses, out = test(layer_loader,device, model, data, split_idx, evaluator)
                train_eval, valid_eval, test_eval = eval_results['train'], eval_results['valid'], eval_results['test']
                train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']

                if valid_eval > best_valid:
                    best_valid = valid_eval
                    best_out = out.cpu().exp()
                # if valid_loss < min_valid_loss:
                #    min_valid_loss = valid_loss
                #    best_out = out.cpu()
                print(
                    f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Loss: {train_loss:.3f}'
                    f'Train: {100 * train_eval:.3f}%, '
                    f'Valid: {100 * valid_eval:.3f}% '
                    f'Test: {100 * test_eval:.3f}%')

                logger.add_result(run, [train_eval, valid_eval, test_eval])

            # -------------saving model ----------------
            if epoch % args.save_step == 0:
                path = "./checkpoints/{}/{}".format(args.dataset, args.model)
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(model.state_dict(), os.path.join(path, "{}_{}_{}.pth".format(run, epoch, valid_eval)))
                print("successfully save {}_{}".format(run, epoch))
        logger.print_statistics(run)
    final_results = logger.print_statistics()
    print('final_results:', final_results)
    param_dict.update(final_results)
    for k, v in param_dict.items():
        if type(v) is list: param_dict.update({k: str(v)})
    pd.DataFrame(param_dict, index=[args.model]).to_csv(result_dir + '/results.csv')

if __name__ == '__main__':
    main()