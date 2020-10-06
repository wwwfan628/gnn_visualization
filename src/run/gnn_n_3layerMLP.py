from src.utils.dataset import load_dataset
from src.utils.train_gnn_n import train_mlp, evaluate_and_classify_nodes_mlp
from src.models.mlp_gnn_n import MLP

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
    g, features, labels, train_mask, valid_mask, test_mask = load_dataset(args)

    # read parameters from config file
    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    h_feats = config['hidden_features']
    in_feats = features.shape[1]
    out_feats = torch.max(labels).item() + 1

    # declarations of variables to save experiment results
    acc = np.zeros(args.exp_times)
    correctly_classified_nodes_list = []

    for i in range(args.exp_times):
        print("********** BUILD NETWORK: {} Experiment **********".format(i+1))
        # build network
        mlp = MLP(3, in_feats, h_feats, out_feats).to(device)

        print("********** TRAIN NETWORK: {} Experiment **********".format(i+1))
        # train network
        _ = train_mlp(mlp, features, labels, train_mask, valid_mask, args)

        print("********** TEST MLP: {} Experiment **********".format(i+1))
        # test with original features
        acc[i], correctly_classified_nodes = evaluate_and_classify_nodes_mlp(mlp, features, labels, test_mask)
        correctly_classified_nodes_list.append(correctly_classified_nodes)
        print("Test accuracy: {:.2f}% !".format(acc[i] * 100))


    print("********** COMPUTE RAVERAGE ACCURACY **********")
    acc_avg = np.mean(acc)
    acc_std = np.std(acc, ddof=1)
    print("Accuracy for {} is {:.2f}+-{:.2f}% !".format(args.dataset, acc_avg*100, acc_std*100))

    print("********** COMPUTE AVERAGE REPEATING RATE **********")
    repeating_rates_list = []
    for i in np.arange(args.exp_times):
        for j in np.arange(i + 1, args.exp_times):
            set_1 = correctly_classified_nodes_list[i]
            set_2 = correctly_classified_nodes_list[j]
            repeating_rates_list.append(len(set_1.intersection(set_2)) * 1.0 / min(len(set_1), len(set_2)))
    rr_avg = np.mean(np.array(repeating_rates_list))
    rr_std = np.std(np.array(repeating_rates_list), ddof=1)
    print("Repeating rate for {} is {:.2f}+-{:.2f}% !".format(args.dataset, rr_avg*100, rr_std*100))

    # save results
    datasets = ["cora", "pubmed", "citeseer", "amazon_photo", "amazon_computers", "coauthors_physics", "coauthors_cs"]
    items = ["acc_mean", "acc_std", "rr_mean", "rr_std"]
    # check if the file exists
    outputs_file = os.path.join(outputs_subdir, 'gnn_n_3layerMLP.csv')
    if os.path.exists(outputs_file):
        # read from file
        results_df = pd.read_csv(outputs_file, index_col=0, header=0)
    else:
        # new array to store results
        # row: dataset    column: item
        results_all_dataset = np.zeros([len(datasets), len(items)])
        results_df = pd.DataFrame(results_all_dataset, index=datasets, columns=items)

    results_df.loc[args.dataset, "acc_mean"] = acc_avg
    results_df.loc[args.dataset, "acc_std"] = acc_std
    results_df.loc[args.dataset, "rr_mean"] = rr_avg
    results_df.loc[args.dataset, "rr_std"] = rr_std
    results_df.to_csv(outputs_file)

    correctly_classified_nodes_list_file = os.path.join(outputs_subdir, 'nodes_list_3layerMLP_' + args.dataset + '.npy')
    with open(correctly_classified_nodes_list_file, 'wb') as f:
        np.save(f, correctly_classified_nodes_list)
        f.close()


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="GNN-N: 100-layer GCN")

    parser.add_argument('--dataset', default='cora', help='choose dataset from: cora, pubmed, citeseer, amazon_photo, amazon_computers, coauthors_cs, coauthors_physics')
    parser.add_argument('--exp_times', type=int, default=10, help='experiment repeating times')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")