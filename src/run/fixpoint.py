from src.utils.dataset import load_dataset
from src.utils.train_model import train_cora_reddit, train_ppi, train_tu
from src.utils.train_model import load_parameters
from src.utils.optimize import optimize_graph_cora_reddit_ppi, optimize_graph_tu, optimize_node_cora_reddit_ppi, optimize_node_tu
from src.models.slp_gcn import SLP_GCN_4node, SLP_GCN_4graph
from src.models.slp import SLP
from src.models.gcn import GCN

import argparse
import torch
import yaml
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("RUNNING ON: {}".format(device))

def main(args):
    # load dataset
    print("********** LOAD DATASET **********")
    if args.dataset in 'cora, reddit-self-loop':
        g, features, labels, train_mask, test_mask = load_dataset(args)
    elif args.dataset == 'ppi':
        train_dataset, train_dataloader, valid_dataloader = load_dataset(args)
    elif args.dataset in 'aids, imdb-binary, reddit-binary':
        statistics, train_dataset, train_dataloader, valid_dataloader = load_dataset(args)

    # build network
    print("********** BUILD NETWORK **********")
    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    h_feats = config['hidden_features']

    if args.dataset in 'cora, reddit-self-loop':
        in_feats = features.shape[1]
        out_feats = torch.max(labels).item() + 1
    elif args.dataset == 'ppi':
        in_feats = train_dataset.features.shape[1]
        out_feats = train_dataset.labels.shape[1]
    elif args.dataset in 'aids, imdb-binary, reddit-binary':
        in_feats = statistics[0]
        out_feats = statistics[1]
        if args.dataset == 'reddit-binary':
            out_feats = out_feats - 1

    if not args.dataset in '':
        slp_gcn = SLP_GCN_4node(in_feats, h_feats, out_feats).to(device)
    else:
        slp_gcn = SLP_GCN_4graph(in_feats, h_feats, out_feats).to(device)

    if args.train:   # need to train the network
        print("********** TRAIN NETWORK **********")
        if args.dataset in 'cora, reddit-self-loop':
            train_cora_reddit(slp_gcn, g, features, labels, train_mask, test_mask, args)
        elif args.dataset == 'ppi':
            train_ppi(slp_gcn, train_dataloader, valid_dataloader, args)
        elif args.dataset in 'aids, imdb-binary, reddit-binary':
            train_tu(slp_gcn, train_dataloader, valid_dataloader, args)

        model_file = 'slp_gcn_parameters_' + args.dataset + '.pkl'
        torch.save(slp_gcn.state_dict(), model_file)
    else:
        path = 'slp_gcn_parameters_' + args.dataset + '.pkl'
        model_file = os.path.join(os.getcwd(), path)
        slp_gcn.load_state_dict(torch.load(model_file, map_location=device))

    # reduce/increase dimension of nodes'features
    print("********** PREPROCESS FEATURES **********")
    slp = SLP(in_feats, h_feats).to(device)
    model_dict = load_parameters(model_file, slp)
    slp.load_state_dict(model_dict)
    slp.eval()
    with torch.no_grad():
        if args.dataset in 'cora, reddit-self-loop':
            features_reduced = slp(features)
        elif args.dataset == 'ppi':
            features = torch.from_numpy(train_dataset.features).to(device)
            features_reduced = slp(features.float())
        elif args.dataset in 'aids, imdb-binary, reddit-binary':
            train_dataset_reduced = train_dataset
            for data in train_dataset_reduced:
                data[0].ndata['feat'] = slp(data[0].ndata['feat'].to(device))

    # GCN
    gcn = GCN(h_feats).to(device)
    model_dict = load_parameters(model_file, gcn)
    gcn.load_state_dict(model_dict)

    # Find fixpoint
    if args.method == 'graph_optimization':
        print("********** OPTIMIZATION ON WHOLE GRAPH **********")
        if args.dataset in 'cora, reddit-self-loop':
            H = optimize_graph_cora_reddit_ppi(gcn, g, features_reduced, args)
        elif args.dataset == 'ppi':
            H = optimize_graph_cora_reddit_ppi(gcn, train_dataset.graph, features_reduced, args)
        elif args.dataset in 'aids, imdb-binary, reddit-binary':
            H, found_indices = optimize_graph_tu(gcn, train_dataset_reduced, args)
    elif args.method == 'node_optimization':
        print("********** OPTIMIZATION ON EACH NODE **********")
        if args.dataset in 'cora, reddit-self-loop':
            H, found_indices = optimize_node_cora_reddit_ppi(gcn, g, features_reduced, args)
        elif args.dataset == 'ppi':
            H, found_indices = optimize_node_cora_reddit_ppi(gcn, train_dataset.graph, features_reduced, args)
        elif args.dataset in 'aids, imdb-binary, reddit-binary':
            H, found_indices = optimize_node_tu(gcn, train_dataset_reduced, args)  # TODO: implement node optimization for tu

    H_file = 'H_' + args.dataset + '_' + args.method + '.pkl'
    torch.save(H, H_file)


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Try to find fixpoint")

    parser.add_argument('dataset', help='choose dataset from: cora, reddit-self-loop, ppi, aids, reddit-binary and imdb-binary')
    parser.add_argument('method', help='choose method from: graph_optimization, node_optimization and newton_method')
    parser.add_argument('--train', action='store_true', help='set true if model needs to be trained, i.e. no checkpoint available')

    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")

