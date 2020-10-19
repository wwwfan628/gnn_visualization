from src.utils.dataset import load_dataset

import argparse
import torch
import yaml
import os
import numpy as np
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("RUNNING ON: {}".format(device))


def main(args):

    # check if 'outputs' directory exist, if not create
    outputs_dir = os.path.join(os.getcwd(), '../outputs')
    outputs_subdir = os.path.join(outputs_dir, 'gnn_n')
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    if not os.path.exists(outputs_subdir):
        os.makedirs(outputs_subdir)

    # load dataset
    print("********** LOAD DATASET **********")
    g, _, _, _, _, _ = load_dataset(args)
    num_nodes = g.number_of_nodes()

    print("********** READ RESULTS FROM FILE **********")
    nodes_list_100layerGCN_file = os.path.join(os.getcwd(), '../outputs/gnn_n/nodes_list_100layerGCN_' + args.dataset + '.npy')
    with open(nodes_list_100layerGCN_file, 'rb') as f:
        nodes_list_100layerGCN = np.load(f, allow_pickle=True)
    f.close()

    nodes_list_3layerMLP_file = os.path.join(os.getcwd(), '../outputs/gnn_n/nodes_list_3layerMLP_' + args.dataset + '.npy')
    with open(nodes_list_3layerMLP_file, 'rb') as f:
        nodes_list_3layerMLP = np.load(f, allow_pickle=True)
    f.close()

    print("********** COMPUTE GNN-N VALUE **********")
    features_probability = np.zeros(num_nodes)
    for i in np.arange(num_nodes):
        correctly_classified_times = 0
        for j in np.arange(args.mlp_exp_times):
            if i in nodes_list_3layerMLP[j]:
                correctly_classified_times += 1
        features_probability[i] = correctly_classified_times * 1.0 / args.mlp_exp_times

    graph_structures_probability = np.zeros(num_nodes)
    for i in np.arange(num_nodes):
        correctly_classified_times = 0
        for j in np.arange(args.gcn_exp_times):
            for k in np.arange(args.gcn_num_random_features):
                if i in nodes_list_100layerGCN[j*args.gcn_num_random_features+k]:
                    correctly_classified_times += 1
        graph_structures_probability[i] = correctly_classified_times * 1.0 / (args.gcn_exp_times * args.gcn_num_random_features)

    gnn_n = np.mean((1-graph_structures_probability)*(1-features_probability))

    # save results
    datasets = ["cora", "pubmed", "citeseer", "amazon_photo", "amazon_computers", "coauthors_physics", "coauthors_cs"]

    # check if the file exists
    outputs_file = os.path.join(outputs_subdir, 'gnn_n.csv')
    if os.path.exists(outputs_file):
        # read from file
        gnn_n_df = pd.read_csv(outputs_file, index_col=0, header=0)
    else:
        # new array to store results
        # row: dataset    column: item
        gnn_n_array = np.zeros([len(datasets), 1])
        gnn_n_df = pd.DataFrame(gnn_n_array, index=datasets, columns=['GNN-N'])

    gnn_n_df.loc[args.dataset, "GNN-N"] = gnn_n
    gnn_n_df.to_csv(outputs_file)



if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="GNN-N")

    parser.add_argument('--dataset', default='cora', help='choose dataset from: cora, pubmed, citeseer, amazon_photo, amazon_computers, coauthors_cs, coauthors_physics')
    parser.add_argument('--mlp_exp_times', type=int, default=10, help='repeating times of 3-layer MLP experiment')
    parser.add_argument('--gcn_exp_times', type=int, default=10, help='repeating times of 100-layer GCN experiment')
    parser.add_argument('--gcn_num_random_features', type=int, default=10, help='how many random features are used for each 100-layer GCN training trial')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")