from src.utils.dataset import load_dataset
from src.utils.train_gnn_n import train_gcn, evaluate_and_classify_nodes_gcn, evaluate_and_classify_nodes_with_random_features_gcn
from src.models.gcn_gnn_n import GCN

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
    acc_original_features = np.zeros(args.exp_times)
    acc_random_features = np.zeros([args.exp_times, args.num_random_features])
    correctly_classified_nodes_original_features_list = []
    correctly_classified_nodes_random_features_list = []

    for i in range(args.exp_times):
        print("********** BUILD NETWORK: {} Experiment **********".format(i+1))
        # build network
        gcn = GCN(100, in_feats, h_feats, out_feats).to(device)

        print("********** TRAIN NETWORK: {} Experiment **********".format(i+1))
        # train network
        _ = train_gcn(gcn, g, features, labels, train_mask, valid_mask, args)

        print("********** TEST WITH ORIGINAL FEATURES: {} Experiment **********".format(i+1))
        # test with original features
        acc_original_features[i], correctly_classified_nodes_original_features = evaluate_and_classify_nodes_gcn(gcn, g, features, labels, test_mask)
        correctly_classified_nodes_original_features_list.append(correctly_classified_nodes_original_features)
        print("Test accuracy with original features: {:.2f}% !".format(acc_original_features[i]*100))

        print("********** TEST WITH RANDOM FEATURES: {} Experiment **********".format(i+1))
        # test with random features
        for j in range(args.num_random_features):
            acc_random_features[i, j], correctly_classified_nodes_random_features = evaluate_and_classify_nodes_with_random_features_gcn(gcn, g, features, labels, test_mask)
            correctly_classified_nodes_random_features_list.append(correctly_classified_nodes_random_features)
            print("Test accuracy with random features {}: {:.2f}% !".format(j+1, acc_random_features[i, j]*100))

    print("********** COMPUTE AVERAGE ACCURACY **********")
    acc_mean_original_features = np.mean(acc_original_features)
    acc_std_original_features = np.std(acc_original_features, ddof=1)
    acc_mean_random_features = np.mean(acc_random_features)
    acc_std_random_features = np.std(acc_random_features, ddof=1)
    print("Average accuracy with original features is {:.2f}+-{:.2f}% !".format(acc_mean_original_features*100,
                                                                                acc_std_original_features*100))
    print("Average accuracy with random features is {:.2f}+-{:.2f}% !".format(acc_mean_random_features * 100,
                                                                                acc_std_random_features * 100))

    print("********** COMPUTE R-RR **********")
    r_rr_list = []
    for i in np.arange(args.exp_times):
        for j in np.arange(args.num_random_features):
            for k in np.arange(j+1, args.num_random_features):
                set_1 = correctly_classified_nodes_random_features_list[i*args.num_random_features+j]
                set_2 = correctly_classified_nodes_random_features_list[i*args.num_random_features+k]
                r_rr_list.append(len(set_1.intersection(set_2)) * 1.0 / min(len(set_1), len(set_2)))
    r_rr_mean = np.mean(np.array(r_rr_list))
    r_rr_std = np.std(np.array(r_rr_list), ddof=1)
    print("R-RR is {:.2f}+-{:.2f}% !".format(r_rr_mean * 100, r_rr_std * 100))

    print("********** COMPUTE RO-RR **********")
    ro_rr_array = np.zeros(args.exp_times)
    for i in np.arange(args.exp_times):
        set_1 = correctly_classified_nodes_original_features_list[i]
        set_2 = correctly_classified_nodes_random_features_list[i*args.num_random_features]
        ro_rr_array[i] = len(set_1.intersection(set_2)) * 1.0 / min(len(set_1), len(set_2))
    ro_rr_mean = np.mean(ro_rr_array)
    ro_rr_std = np.std(ro_rr_array, ddof=1)
    print("RO-RR is {:.2f}+-{:.2f}% !".format(ro_rr_mean * 100, ro_rr_std * 100))

    print("********** COMPUTE TT-RR **********")
    tt_rr_array = np.eye(args.exp_times)
    for i in np.arange(args.exp_times):
        for j in np.arange(i+1, args.exp_times):
            set_1 = correctly_classified_nodes_random_features_list[i*args.num_random_features]
            set_2 = correctly_classified_nodes_random_features_list[j*args.num_random_features]
            tt_rr_array[i, j] = len(set_1.intersection(set_2)) * 1.0 / min(len(set_1), len(set_2))
            tt_rr_array[j, i] = len(set_1.intersection(set_2)) * 1.0 / min(len(set_1), len(set_2))

    # save results
    datasets = ["cora", "pubmed", "citeseer", "amazon_photo", "amazon_computers", "coauthors_physics", "coauthors_cs"]
    items = ["acc mean original features", "acc std original features", "acc mean random features",
             "acc std random features", "r_rr mean", "r_rr std", "ro_rr mean", "ro_rr std"]

    # check if the file exists
    outputs_file = os.path.join(outputs_subdir, 'gnn_n_100layerGCN.csv')
    if os.path.exists(outputs_file):
        # read from file
        results_df = pd.read_csv(outputs_file, index_col=0, header=0)
    else:
        # new array to store results
        # row: dataset    column: item
        results_all_dataset = np.zeros([len(datasets), len(items)])
        results_df = pd.DataFrame(results_all_dataset, index=datasets, columns=items)

    results_df.loc[args.dataset, "acc mean original features"] = acc_mean_original_features
    results_df.loc[args.dataset, "acc std original features"] = acc_std_original_features
    results_df.loc[args.dataset, "acc mean random features"] = acc_mean_random_features
    results_df.loc[args.dataset, "acc std random features"] = acc_std_random_features
    results_df.loc[args.dataset, "r_rr mean"] = r_rr_mean
    results_df.loc[args.dataset, "r_rr std"] = r_rr_std
    results_df.loc[args.dataset, "ro_rr mean"] = ro_rr_mean
    results_df.loc[args.dataset, "ro_rr std"] = ro_rr_std
    results_df.to_csv(outputs_file)

    tt_rr_file = os.path.join(outputs_subdir, 'tt_rr_'+args.dataset+'.npy')
    with open(tt_rr_file, 'wb') as f:
        np.save(f, tt_rr_array)
        f.close()

    correctly_classified_nodes_random_features_list_file = os.path.join(outputs_subdir, 'nodes_list_100layerGCN_' + args.dataset + '.npy')
    with open(correctly_classified_nodes_random_features_list_file, 'wb') as f:
        np.save(f, correctly_classified_nodes_random_features_list)
        f.close()

if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="GNN-N: 100-layer GCN")

    parser.add_argument('--dataset', default='cora', help='choose dataset from: cora, pubmed, citeseer, amazon_photo, amazon_computers, coauthors_cs, coauthors_physics')
    parser.add_argument('--exp_times', type=int, default=10, help='experiment repeating times')
    parser.add_argument('--num_random_features', type=int, default=10, help='how many random features should be used for each experiment')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")