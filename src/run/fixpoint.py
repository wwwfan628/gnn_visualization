from src.utils.dataset import load_dataset
from src.utils.train_model import train_cora_reddit, train_ppi, train_tu
from src.utils.train_model import load_parameters
from src.utils.optimize import optimize_graph_cora_reddit_ppi, optimize_graph_tu
from src.models.slp_gcn import SLP_GCN_4node, SLP_GCN_4graph
from src.models.slp import SLP
from src.models.gcn import GCN

import argparse
import torch
import yaml
import os

def main(args):

    # load dataset
    if args.dataset == 'cora' or 'reddit':
        g, features, labels, train_mask, test_mask = load_dataset(args.dataset)
    elif args.dataset == 'ppi':
        train_dataset, train_dataloader, valid_dataloader = load_dataset(args.dataset)
    elif args.dataset == 'tu':
        statistics, train_dataset, train_dataloader, valid_dataloader = load_dataset(args.dataset)

    # build network
    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    h_feats = config['hidden_features']

    if args.dataset == 'cora' or 'reddit':
        in_feats = features.shape[1]
        out_feats = torch.max(labels).item() + 1
    elif args.dataset == 'ppi':
        in_feats = train_dataset.features.shape[1]
        out_feats = train_dataset.labels.shape[1]
    elif args.dataset == 'tu':
        in_feats = statistics[0]
        out_feats = statistics[1]

    if args.dataset != 'tu':
        slp_gcn = SLP_GCN_4node(in_feats, h_feats, out_feats)
    else:
        slp_gcn = SLP_GCN_4graph(in_feats, h_feats, out_feats)

    if args.train:   # need to train the network
        if args.dataset == 'cora' or 'reddit':
            train_cora_reddit(slp_gcn, g, features, labels, train_mask, test_mask, args)
        elif args.dataset == 'ppi':
            train_ppi(slp_gcn, train_dataloader, valid_dataloader, args)
        elif args.dataset == 'tu':
            train_tu(slp_gcn, train_dataloader, valid_dataloader, args)

        model_file = 'slp_gcn_parameters_' + args.dataset + '.pkl'
        torch.save(slp_gcn.state_dict(), model_file)
    else:
        model_file = 'slp_gcn_parameters_' + args.dataset + '.pkl'
        slp_gcn.load_state_dict(model_file)

    # reduce/increase dimension of nodes'features
    slp = SLP(in_feats, h_feats)
    model_dict = load_parameters(model_file, slp)
    slp.load_state_dict(model_dict)
    slp.eval()
    with torch.no_grad():
        if args.dataset == 'cora' or 'reddit':
            features_reduced = slp(features)
        elif args.dataset == 'ppi':
            features_reduced = slp(train_dataset.features)
        elif args.dataset == 'tu':
            train_dataset_reduced = train_dataset
            for data in train_dataset_reduced:
                data[0].ndata['feat'] = slp(data[0].ndata['feat'])

    # GCN
    gcn = GCN(h_feats)
    model_dict = load_parameters(model_file, gcn)
    gcn.load_state_dict(model_dict)

    # Find fixpoint
    if args.method == 'graph_optimization':
        if args.dataset == 'cora' or 'reddit':
            H = optimize_graph_cora_reddit_ppi(gcn, g, features_reduced, args)
        elif args.dataset == 'ppi':
            H = optimize_graph_cora_reddit_ppi(gcn, train_dataset.graph, features_reduced, args)
        elif args.dataset == 'tu':
            H , found_indices = optimize_graph_tu(gcn, train_dataset_reduced, args)


    H_file = 'H_' + args.dataset + '.pkl'
    torch.save(H, H_file)


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Try to find fixpoint")

    parser.add_argument('dataset', help='choose dataset from: cora, reddit, ppi and tu')
    parser.add_argument('method', help='choose method from: graph_optimization, node_optimization and newton_method')
    parser.add_argument('--train', action='store_true', help='set true if model needs to be trained, i.e. no checkpoint available')

    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")

