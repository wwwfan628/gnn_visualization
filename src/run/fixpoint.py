from src.utils.dataset import load_dataset
from src.utils.train_model import train_cora_reddit, train_ppi, train_tu, evaluate_tu, evaluate_ppi, evaluate_cora_reddit
from src.utils.train_model import load_parameters
from src.utils.optimize import optimize_graph_cora_reddit_ppi, optimize_graph_tu, optimize_node_cora_reddit_ppi, optimize_node_tu
from src.utils.newton_method import newton_method_cora_reddit_ppi, newton_method_tu
from src.utils.broyden_method import broyden_method_cora_reddit_ppi, broyden_method_tu
from src.models.slp_gcn import SLP_GCN_4node, SLP_GCN_4graph
from src.models.slp import SLP
from src.models.gcn import GCN
from src.models.last_layer import Last_Layer_4graph, Last_Layer_4node

import argparse
import torch
import yaml
import os
import numpy as np
import random
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("RUNNING ON: {}".format(device))

def fix_random_seed(seed=0):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main(args):

    # fix random seed if True
    if args.fix_random:
        fix_random_seed()

    # check if 'outputs' and 'checkpoints' directory exist, if not create
    outputs_dir = os.path.join(os.getcwd(), '../outputs')
    checkpoints_dir = os.path.join(os.getcwd(), '../checkpoints')
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # load dataset
    print("********** LOAD DATASET **********")
    if args.dataset in 'cora, reddit-self-loop':
        g, features, labels, train_mask, test_mask = load_dataset(args)
    elif args.dataset == 'ppi':
        train_dataset, valid_dataset, train_dataloader, valid_dataloader = load_dataset(args)
    elif 'tu' in args.dataset:
        statistics, train_dataset, valid_dataset, train_dataloader, valid_dataloader = load_dataset(args)

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
    elif 'tu' in args.dataset:
        in_feats = statistics[0]
        out_feats = statistics[1].item()

    if 'tu' not in args.dataset:
        slp_gcn = SLP_GCN_4node(in_feats, h_feats, out_feats).to(device)
    else:
        slp_gcn = SLP_GCN_4graph(in_feats, h_feats, out_feats).to(device)

    if args.train:   # need to train the network
        print("********** TRAIN NETWORK **********")
        if args.dataset in 'cora, reddit-self-loop':
            train_cora_reddit(slp_gcn, g, features, labels, train_mask, test_mask, args)
        elif args.dataset == 'ppi':
            train_ppi(slp_gcn, train_dataloader, valid_dataloader, args)
        elif 'tu' in args.dataset:
            train_tu(slp_gcn, train_dataloader, valid_dataloader, args)

        checkpoint_path = '../checkpoints/slp_gcn_' + args.dataset + '.pkl'
        checkpoint_file = os.path.join(os.getcwd(), checkpoint_path)
        torch.save(slp_gcn.state_dict(), checkpoint_file)

    else:
        checkpoint_path = '../checkpoints/slp_gcn_' + args.dataset + '.pkl'
        checkpoint_file = os.path.join(os.getcwd(), checkpoint_path)
        slp_gcn.load_state_dict(torch.load(checkpoint_file, map_location=device))

    # reduce/increase dimension of training set
    print("********** PREPROCESS FEATURES FOR TRAINING SET **********")
    slp = SLP(in_feats, h_feats).to(device)
    model_dict = load_parameters(checkpoint_file, slp)
    slp.load_state_dict(model_dict)
    slp.eval()
    with torch.no_grad():
        if args.dataset in 'cora, reddit-self-loop':
            features_reduced = slp(features)
        elif args.dataset == 'ppi':
            features = torch.from_numpy(train_dataset.features).to(device)
            features_reduced = slp(features.float())
        elif 'tu' in args.dataset:
            train_dataset_reduced = train_dataset
            for data in train_dataset_reduced:
                data[0].ndata['feat'] = slp(data[0].ndata['feat'].float().to(device))

    # GCN
    gcn = GCN(h_feats).to(device)
    model_dict = load_parameters(checkpoint_file, gcn)
    gcn.load_state_dict(model_dict)

    # Find fixpoint for training set
    print("********** FIND FIXPOINT FOR TRAINING SET **********")
    if args.method == 'graph_optimization':
        print("********** OPTIMIZATION ON WHOLE GRAPH **********")
        if args.dataset in 'cora, reddit-self-loop':
            H, min_cost_func = optimize_graph_cora_reddit_ppi(gcn, g, features_reduced, args)
        elif args.dataset == 'ppi':
            H, min_cost_func = optimize_graph_cora_reddit_ppi(gcn, train_dataset.graph, features_reduced, args)
        elif 'tu' in args.dataset:
            H, found_indices, min_cost_func = optimize_graph_tu(gcn, train_dataset_reduced, args)
    elif args.method == 'node_optimization':
        print("********** OPTIMIZATION ON EACH NODE **********")
        if args.dataset in 'cora, reddit-self-loop':
            H, found_indices = optimize_node_cora_reddit_ppi(gcn, g, features_reduced, args)
        elif args.dataset == 'ppi':
            H, found_indices = optimize_node_cora_reddit_ppi(gcn, train_dataset.graph, features_reduced, args)
        elif 'tu' in args.dataset:
            H, found_indices = optimize_node_tu(gcn, train_dataset_reduced, args)
    elif args.method == 'newton_method':
        print("********** NEWTON'S METHOD **********")
        if args.dataset in 'cora, reddit-self-loop':
            H, min_cost_func = newton_method_cora_reddit_ppi(gcn, g, features_reduced, args)
        elif args.dataset == 'ppi':
            H, min_cost_func = newton_method_cora_reddit_ppi(gcn, train_dataset.graph, features_reduced, args)
        elif 'tu' in args.dataset:
            H, found_indices, min_cost_func = newton_method_tu(gcn, train_dataset_reduced, args)
    elif 'broyden' in args.method:
        print("********** BROYDEN'S METHOD **********")
        if args.dataset in 'cora, reddit-self-loop':
            H, min_cost_func = broyden_method_cora_reddit_ppi(gcn, g, features_reduced, args)
        elif args.dataset == 'ppi':
            H, min_cost_func = broyden_method_cora_reddit_ppi(gcn, train_dataset.graph, features_reduced, args)
        elif 'tu' in args.dataset:
            H, found_indices, min_cost_func = broyden_method_tu(gcn, train_dataset_reduced, args)

    # Save result
    H_path = '../outputs/H_' + args.dataset + '_' + args.method + '.pkl'
    H_file = os.path.join(os.getcwd(), H_path)
    torch.save(H, H_file)
    cost_func_path = '../outputs/cost_func_' + args.dataset + '_' + args.method + '.pkl'
    cost_func_file = os.path.join(os.getcwd(), cost_func_path)
    torch.save(min_cost_func, cost_func_file)
    if 'tu' in args.dataset:
        indices_path = '../outputs/indices_' + args.dataset + '_' + args.method + '.pkl'
        indices_file = os.path.join(os.getcwd(), indices_path)
        torch.save(found_indices, indices_file)

    # Test fixpoint's performance in classification
    if args.test:
        print("********** TEST OF FIXPOINT **********")
        # Reduce/increase dimension of validation set
        print("********** PREPROCESS FEATURES FOR VALIDATION SET **********")
        slp.eval()
        with torch.no_grad():
            if args.dataset == 'ppi':
                features_val = torch.from_numpy(valid_dataset.features).to(device)
                features_reduced_val = slp(features_val.float())
            elif 'tu' in args.dataset:
                valid_dataset_reduced = valid_dataset
                for data in valid_dataset_reduced:
                    data[0].ndata['feat'] = slp(data[0].ndata['feat'].float().to(device))

        # Find fixpoint for validation set
        print("********** FIND FIXPOINT FOR VALIDATION SET **********")
        if args.method == 'graph_optimization':
            print("********** OPTIMIZATION ON WHOLE GRAPH **********")
            if args.dataset == 'ppi':
                H_val, min_cost_func_val = optimize_graph_cora_reddit_ppi(gcn, valid_dataset.graph, features_reduced_val, args)
            elif 'tu' in args.dataset:
                H_val, found_indices_val, min_cost_func_val = optimize_graph_tu(gcn, valid_dataset_reduced, args)
        elif args.method == 'node_optimization':
            print("********** OPTIMIZATION ON EACH NODE **********")
            if args.dataset == 'ppi':
                H_val, found_indices_val = optimize_node_cora_reddit_ppi(gcn, valid_dataset.graph, features_reduced_val, args)
            elif 'tu' in args.dataset:
                H_val, found_indices_val = optimize_node_tu(gcn, valid_dataset_reduced, args)
        elif args.method == 'newton_method':
            print("********** NEWTON'S METHOD **********")
            if args.dataset == 'ppi':
                H_val, min_cost_func_val = newton_method_cora_reddit_ppi(gcn, valid_dataset.graph, features_reduced_val, args)
            elif 'tu' in args.dataset:
                H_val, found_indices_val, min_cost_func_val = newton_method_tu(gcn, valid_dataset_reduced, args)
        elif 'broyden' in args.method:
            print("********** BROYDEN'S METHOD **********")
            if args.dataset == 'ppi':
                H_val, min_cost_func_val = broyden_method_cora_reddit_ppi(gcn, valid_dataset.graph, features_reduced_val, args)
            elif 'tu' in args.dataset:
                H_val, found_indices_val, min_cost_func_val = broyden_method_tu(gcn, valid_dataset_reduced, args)

        # Build last layer
        print("********** BUILD LAST LAYER **********")
        if 'tu' in args.dataset:
            last_layer = Last_Layer_4graph(h_feats, out_feats)
            model_dict = load_parameters(checkpoint_file, last_layer)
            last_layer.load_state_dict(model_dict)
        else:
            last_layer = Last_Layer_4node(h_feats, out_feats)
            model_dict = load_parameters(checkpoint_file, last_layer)
            last_layer.load_state_dict(model_dict)

        # Train last layer
        print("********** TRAIN LAST LAYER **********")
        if 'tu' in args.dataset:
            for graph_idx, (graph, graph_label) in enumerate(train_dataset):
                graph.ndata['feat'] = H[graph_idx]
            for graph_idx, (graph, graph_label) in enumerate(valid_dataset):
                graph.ndata['feat'] = H_val[graph_idx]
            train_tu(last_layer, train_dataloader, valid_dataloader, args)
        elif args.dataset in 'cora, reddit-self-loop':
            train_cora_reddit(slp_gcn, g, H, labels, train_mask, test_mask, args)
        elif args.dataset == 'ppi':
            train_dataset.features = H
            valid_dataset.features = H_val
            train_ppi(slp_gcn, train_dataloader, valid_dataloader, args)

if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Try to find fixpoint")

    parser.add_argument('dataset', help='choose dataset from: cora, reddit-self-loop, ppi, aids, reddit-binary and imdb-binary')
    parser.add_argument('method', help='choose method from: graph_optimization, node_optimization and newton_method')
    parser.add_argument('--train', action='store_true', help='set true if model needs to be trained, i.e. no checkpoint available')
    parser.add_argument('--fix_random', action='store_true', help='set true if repeatability required')
    parser.add_argument('--test', action='store_true', help='set true to test fixpoint\'s performance in classification task' )
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")

