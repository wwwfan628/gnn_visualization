from src.utils.dataset import load_dataset
from src.utils.train_fixedpoint import train_gcn
from src.models.gcn_fixedpoint import GCN

import argparse
import torch
import yaml
import os
import pandas as pd
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("RUNNING ON: {}".format(device))


def main(args):

    # check if 'outputs' directory exist, if not create
    outputs_dir = os.path.join(os.getcwd(), '../outputs')
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

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

    # array to save
    acc_array = np.zeros(args.exp_times)

    for i in range(args.exp_times):
        print("********** BUILD NETWORK: {} Iteration **********".format(i))
        # build network
        gcn = GCN(in_feats, h_feats, out_feats).to(device)

        print("********** TRAIN NETWORK: {} Iteration **********".format(i))
        # train network
        best_acc = train_gcn(gcn, g, features, labels, train_mask, test_mask, args)

        acc_array[i] = best_acc  # store in the acc_array

    acc_current_dataset = np.mean(acc_array)*100
    print("Average accuracy for dataset {} is {}% !".format(args.dataset, acc_current_dataset))

    # save results
    if args.fixpoint_loss:
        models = ["GCN", "SSE", "GCN trained with joint loss"]
        datasets = ["cora", "pubmed", "citeseer"]
        # check if the file exists
        outputs_file = os.path.join(os.getcwd(), '../outputs/fixedpoint.csv')
        if os.path.exists(outputs_file):
            # read from file
            acc_df = pd.read_csv(outputs_file, index_col=0, header=0)
        else:
            # new array to store accracy
            # row: dataset    column: model
            acc_all_dataset = np.array([[81.5, 79.4, 0], [79.0, 75.8, 0], [70.3, 72.5, 0]])
            acc_df = pd.DataFrame(acc_all_dataset, index=datasets, columns=models)

        acc_df.loc[args.dataset, "GCN trained with joint loss"] = acc_current_dataset
        acc_df.to_csv(outputs_file)


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Fixed Point")

    parser.add_argument('--dataset', default='cora', help='choose dataset from: cora, pubmed, citeseer')
    parser.add_argument('--exp_times', type=int, default=1, help='experiment repeating times')
    parser.add_argument('--fixpoint_loss', action='store_true', help='if true add fixpoint loss, else only classification loss')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")

