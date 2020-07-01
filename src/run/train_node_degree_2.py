from src.utils.dataset import load_dataset
from src.utils.train_embedding_id import train_citation
from src.models.gcn_embedding_id import GCN_1Layer,GCN_2Layers,GCN_3Layers,GCN_4Layers,GCN_5Layers,GCN_6Layers,GCN_7Layers,GCN_8Layers,GCN_9Layers,GCN_10Layers

import argparse
import torch
import os
import yaml
import pandas as pd
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("RUNNING ON: {}".format(device))


def main(args):

    # check if 'outputs' and 'checkpoints' directory exist, if not create
    outputs_dir = os.path.join(os.getcwd(), '../outputs')
    outputs_subdir = os.path.join(outputs_dir, args.info)
    checkpoints_dir = os.path.join(os.getcwd(), '../checkpoints')
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    if not os.path.exists(outputs_subdir):
        os.makedirs(outputs_subdir)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    if args.dataset in 'cora, citeseer, pubmed':
        # load dataset
        print("********** LOAD DATASET **********")
        g, features, labels, train_mask, valid_mask, test_mask = load_dataset(args)

        # modify dataset: labels-->degree  features-->constant 1
        features = torch.rand((g.number_of_nodes(),1))
        labels = torch.sum(g.adjacency_matrix(transpose=True).to_dense(),dim=1).long()-1
        labels[labels>=9] = 9

        # prepare to build network
        path = '../configs/' + args.dataset + '.yaml'
        config_file = os.path.join(os.getcwd(), path)
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        h_feats = config['hidden_features']
        in_feats = features.shape[1]
        out_feats = torch.max(labels).item()+1

        # result before averaging over experiments'iterations
        # store 'accracy' for different iterations/models
        # 1st dim.:repeating time  2nd dim.:model
        accuracy = np.zeros([args.repeat_times, args.max_gcn_layers])

        for repeat_time in np.arange(args.repeat_times):
            for gcn_model_layer in np.arange(1, args.max_gcn_layers + 1):
                print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time + 1))
                print("********** GCN MODEL: GCN_{}layers **********".format(gcn_model_layer))
                print("********** BUILD GCN NETWORK**********")
                if gcn_model_layer == 1:
                    gcn = GCN_1Layer(in_feats, out_feats).to(device)
                elif gcn_model_layer == 2:
                    gcn = GCN_2Layers(in_feats, h_feats, out_feats).to(device)
                elif gcn_model_layer == 3:
                    gcn = GCN_3Layers(in_feats, h_feats, out_feats).to(device)
                elif gcn_model_layer == 4:
                    gcn = GCN_4Layers(in_feats, h_feats, out_feats).to(device)
                elif gcn_model_layer == 5:
                    gcn = GCN_5Layers(in_feats, h_feats, out_feats).to(device)
                elif gcn_model_layer == 6:
                    gcn = GCN_6Layers(in_feats, h_feats, out_feats).to(device)
                elif gcn_model_layer == 7:
                    gcn = GCN_7Layers(in_feats, h_feats, out_feats).to(device)
                elif gcn_model_layer == 8:
                    gcn = GCN_8Layers(in_feats, h_feats, out_feats).to(device)
                elif gcn_model_layer == 9:
                    gcn = GCN_9Layers(in_feats, h_feats, out_feats).to(device)
                elif gcn_model_layer == 10:
                    gcn = GCN_10Layers(in_feats, h_feats, out_feats).to(device)

                print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time + 1))
                print("********** GCN MODEL: GCN_{}layer **********".format(gcn_model_layer))
                print("********** TRAIN GCN NETWORK **********")
                acc = train_citation(gcn, g, features, labels, train_mask, test_mask, args)
                accuracy[repeat_time, gcn_model_layer - 1] = acc

                models = []  # dataframe column index
                for gcn_model_layer in np.arange(1, args.max_gcn_layers + 1):
                    if gcn_model_layer == 1:
                        model_name = 'gcn_1layer'
                    else:
                        model_name = 'gcn_' + str(gcn_model_layer) + 'layers'
                    models.append(model_name)
                accuracy_df = pd.DataFrame(np.mean(accuracy, axis=0), index=models, columns=['accuracy'])

                save_subpath = 'accuracy_' + args.dataset + '_' + str(args.knn) + '-nn' + '.csv'
                save_path = os.path.join(outputs_subdir, save_subpath)
                accuracy_df.to_csv(save_path)

                checkpoint_file_name = 'gcn_' + str(gcn_model_layer) + 'layers_' + args.dataset + '.pkl'
                checkpoint_file = os.path.join(checkpoints_dir, checkpoint_file_name)
                torch.save(gcn.state_dict(), checkpoint_file)


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Try to find fixpoint")

    parser.add_argument('--dataset', default='cora', help='choose dataset from: cora, pubmed, citeseer, ppi')
    parser.add_argument('--info', default='degree_random', help='choose the information to recover')
    parser.add_argument('--regression_model', default='mlp', help='choose model structure from: slp, mlp')
    parser.add_argument('--regression_metric', default='cos', help='choose metric for regression and nearest neighbour from: l2 or cos')
    parser.add_argument('--knn', type=int, default=1, help='method to find the corresponding node among neighboring nodes after recovery, k=1,2 or 3')
    parser.add_argument('--repeat_times', type=int, default=10, help='experiment repeating times for single layer')
    parser.add_argument('--max_gcn_layers', type=int, default=10, help='the maxmal gcn models\'s layer, not larger than 8')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")