from src.utils.dataset import load_dataset
from src.utils.train_gnn_n import train_mlp, evaluate_and_classify_nodes_mlp
from src.models.mlp_gnn_n import MLP

import argparse
import torch
import yaml
import os
import numpy as np

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
        print("Test accuracy: {:.2f} !".format(acc[i] * 100))

    # save results
    with open('../outputs/gnn_n/3layersMLP_'+ args.dataset+'.npy', 'wb') as f:
        np.save(f, acc)
        np.save(f, correctly_classified_nodes_list)
        f.close()


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="GNN-N: 100-layer GCN")

    parser.add_argument('--dataset', default='cora', help='choose dataset from: cora, pubmed, citeseer')
    parser.add_argument('--exp_times', type=int, default=10, help='experiment repeating times')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")