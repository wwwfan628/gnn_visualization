from src.utils.dataset import load_dataset
from src.utils.train_embedding_id import train_citation, train_ppi, train_reg_citation,train_reg_ppi
from src.models.gcn_embedding_id import GCN_2Layers, GCN_3Layers, GCN_4Layers, GCN_5Layers, GCN_6Layers
from src.models.regression_embedding_id import MLP, SLP

import argparse
import torch
import os
import numpy as np
import yaml
import pandas as pd


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

    # load dataset
    print("********** LOAD DATASET **********")
    if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed':
        g, features, labels, train_mask, test_mask = load_dataset(args)
    elif args.dataset == 'ppi':
        train_dataset, valid_dataset, train_dataloader, valid_dataloader = load_dataset(args)

    # build network
    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    h_feats = config['hidden_features']

    if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed':
        in_feats = features.shape[1]
        out_feats = torch.max(labels).item() + 1
    elif args.dataset == 'ppi':
        in_feats = train_dataset.features.shape[1]
        out_feats = train_dataset.labels.shape[1]

    # store embedding id rates for different models, row:model  column:embedding layer
    models = ['gcn_2layers', 'gcn_3layers', 'gcn_4layers', 'gcn_5layers', 'gcn_6layers']
    recoverability_rates_df = pd.DataFrame(np.zeros([6,5]), index=np.arange(1,7), columns=models)
    node_repeatability_rates_df = pd.DataFrame(np.zeros([6, 5]), index=np.arange(1, 7), columns=models)
    for gcn_model_layer in np.arange(2, 7):
        # store embedding id rates for one model, row:embedding layer  column:experiment iteration time
        recoverability_rates = np.zeros([6,5])
        # node repeatablity rates for one model
        node_repeatablity_rates = np.zeros([6,5])
        for repeat_time in np.arange(args.repeat_times):
            print("********** GCN MODEL: GCN_{}layer **********".format(gcn_model_layer))
            print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time+1))
            print("********** BUILD GCN NETWORK **********")
            if gcn_model_layer == 2:
                gcn = GCN_2Layers(in_feats, h_feats, out_feats).to(device)
            elif gcn_model_layer == 3:
                gcn = GCN_3Layers(in_feats, h_feats, out_feats).to(device)
            elif gcn_model_layer == 4:
                gcn = GCN_4Layers(in_feats, h_feats, out_feats).to(device)
            elif gcn_model_layer == 5:
                gcn = GCN_5Layers(in_feats, h_feats, out_feats).to(device)
            elif gcn_model_layer == 6:
                gcn = GCN_6Layers(in_feats, h_feats, out_feats).to(device)

            print("********** GCN MODEL: GCN_{}layer **********".format(gcn_model_layer))
            print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time+1))
            print("********** TRAIN GCN NETWORK **********")
            if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed':
                train_citation(gcn, g, features, labels, train_mask, test_mask, args)
            elif args.dataset == 'ppi':
                train_ppi(gcn, train_dataloader, valid_dataloader, args)

            checkpoint_path = '../checkpoints/gcn_' + str(gcn_model_layer) + 'layers_' + args.dataset + '.pkl'
            checkpoint_file = os.path.join(os.getcwd(), checkpoint_path)
            torch.save(gcn.state_dict(), checkpoint_file)

            print("********** GCN MODEL: GCN_{}layer **********".format(gcn_model_layer))
            print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time+1))
            print("********** BUILD REGRESSION MODEL TO RECOVER INPUT FROM EMBEDDING **********")
            for intermediate_layer in np.arange(1, gcn_model_layer + 1):
                print("********** INTERMEDIATE LAYER: {} **********".format(intermediate_layer))
                if args.dataset in 'cora, citeseer, pubmed':
                    embedding = gcn(g, features)[-intermediate_layer].clone().detach().to(device)
                    nodes_degree = torch.zeros([g.number_of_nodes(),1])
                    for node_ind in range(g.number_of_nodes()):
                        nodes_degree[node_ind] = len(g.adjacency_matrix(transpose=True)[node_ind].to_dense() == 1)
                    reg_in = embedding.shape[1]
                    reg_out = 1
                    reg_h = config['regression_hidden_features_degree']
                elif args.dataset == 'ppi': #TODO:IMPLEMENTATION
                    for data in train_dataset:
                        data[0].ndata['embedding'] = gcn(data[0], data[0].ndata['feat'].float(), args)[1].detach()
                    for data in valid_dataset:
                        data[0].ndata['embedding'] = gcn(data[0], data[0].ndata['feat'].float(), args)[1].detach()
                    reg_in = data[0].ndata['embedding'].shape[1]
                    reg_out = data[0].ndata['feat'].shape[1]
                    reg_h = config['regression_hidden_features_degree']

                if args.regression_model == 'mlp':
                    reg = MLP(reg_in, reg_h, reg_out)
                elif args.regression_model == 'slp':
                    reg = SLP(reg_in, reg_out)

                print("********** GCN MODEL: GCN_{}layer **********".format(gcn_model_layer))
                print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time + 1))
                print("********** TRAIN REGRESSION MODEL TO RECOVER INPUT FROM EMBEDDING **********")
                print("********** INTERMEDIATE LAYER: {} **********".format(intermediate_layer))
                if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed':
                    train_reg_citation(reg, embedding, nodes_degree, train_mask, test_mask, args)
                elif args.dataset == 'ppi':
                    train_reg_ppi(reg, train_dataloader, valid_dataloader, args)

                print("********** GCN MODEL: GCN_{}layer **********".format(gcn_model_layer))
                print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time + 1))
                print("********** INTERMEDIATE LAYER: {} **********".format(intermediate_layer))
                print("********** COMPUTE RECOVERABILITY **********")
                if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed':
                    reg_output = reg(embedding)
                    if intermediate_layer == 1:
                        recovered_node_1layer = set(torch.arange(g.number_of_nodes())[torch.squeeze(reg_output==nodes_degree)])
                    recovered_node_current_layer = set(torch.arange(g.number_of_nodes())[torch.squeeze(reg_output==nodes_degree)])
                    num_recoverable = len(recovered_node_current_layer)
                    num_repeatable = len(recovered_node_current_layer.intersection(recovered_node_1layer))
                    recoverability_rate = num_recoverable * 1.0 / g.number_of_nodes()
                    repeatability_rate = num_repeatable * 1.0 / g.number_of_nodes()
                print('identifiability_rate = {} %'.format(recoverability_rate * 100))
                print('node_repeatability_rate = {} %'.format(repeatability_rate * 100))
                recoverability_rates[intermediate_layer - 1, repeat_time] = recoverability_rate
                node_repeatablity_rates[intermediate_layer - 1, repeat_time] = repeatability_rate

        df_column_ind = models[gcn_model_layer-2]
        recoverability_rates_df[df_column_ind] = np.mean(recoverability_rates, axis=1)
        node_repeatability_rates_df[df_column_ind] = np.mean(node_repeatablity_rates, axis=1)

    save_subpath = 'recoverability_rates_' + args.dataset + '.csv'
    save_path = os.path.join(outputs_subdir, save_subpath)
    recoverability_rates_df.to_csv(save_path)
    save_subpath = 'repeatability_rates_' + args.dataset + '.csv'
    save_path = os.path.join(outputs_subdir, save_subpath)
    node_repeatability_rates_df.to_csv(save_path)


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Try to find fixpoint")

    parser.add_argument('--dataset', default='cora', help='choose dataset from: cora, pubmed, citeseer, ppi')
    parser.add_argument('--regression_model', default='mlp', help='choose model structure from: slp, mlp or none')
    parser.add_argument('--regression_metric', default='cos', help='choose metric for regression and nearest neighbour from: l2 or cos')
    parser.add_argument('--repeat_times', type=int, default=5, help='experiment repeating times for single layer')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")