from src.utils.dataset import load_dataset
from src.utils.train_gnn_n import train_gcn, evaluate_and_classify_nodes_gcn, evaluate_and_classify_nodes_with_random_features_gcn
from src.models.gcn_gnn_n import GCN

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
    acc_original_features = np.zeros(args.exp_times)
    acc_random_features = np.zeros([args.exp_times, args.num_random_features])
    correctly_classified_nodes_original_features_list = []
    correctly_classified_nodes_random_features_list = []

    for i in range(args.exp_times):
        print("********** BUILD NETWORK: {} Experiment **********".format(i))
        # build network
        gcn = GCN(100, in_feats, h_feats, out_feats).to(device)

        print("********** TRAIN NETWORK: {} Experiment **********".format(i))
        # train network
        _ = train_gcn(gcn, g, features, labels, train_mask, valid_mask, args)

        print("********** TEST WITH ORIGINAL FEATURES: {} Experiment **********".format(i))
        # test with original features
        acc_original_features[i], correctly_classified_nodes_original_features = evaluate_and_classify_nodes_gcn(gcn, g, features, labels, test_mask)
        correctly_classified_nodes_original_features_list.append(correctly_classified_nodes_original_features)

        print("********** TEST WITH RANDOM FEATURES: {} Experiment **********".format(i))
        # test with random features
        for j in range(args.num_random_features):
            acc_random_features[i, j], correctly_classified_nodes_random_features = evaluate_and_classify_nodes_with_random_features_gcn(gcn, g, features, labels, test_mask)
            correctly_classified_nodes_random_features_list.append(correctly_classified_nodes_random_features)

    # save results
    with open('../outputs/gnn_n/100layersGCN_'+ args.dataset+'.npy', 'wb') as f:
        np.save(f, acc_original_features)
        np.save(f, acc_random_features)
        np.save(f, correctly_classified_nodes_original_features_list)
        np.save(f, correctly_classified_nodes_random_features_list)
        f.close()


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="GNN-N: 100-layer GCN")

    parser.add_argument('--dataset', default='cora', help='choose dataset from: cora, pubmed, citeseer')
    parser.add_argument('--exp_times', type=int, default=10, help='experiment repeating times')
    parser.add_argument('--num_random_features', type=int, default=10, help='how many random features should be used for each experiment')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")