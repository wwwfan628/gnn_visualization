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
    checkpoints_dir = os.path.join(os.getcwd(), '../checkpoints')
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
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
    embedding_id_rates_df = pd.DataFrame(np.zeros([6,5]), index=np.arange(1,7), columns=models)
    node_repeatability_rates_df = pd.DataFrame(np.zeros([6, 5]), index=np.arange(1, 7), columns=models)
    for gcn_model_layer in np.arange(2, 7):
        # store embedding id rates for one model, row:embedding layer  column:experiment iteration time
        embedding_id_rates = np.zeros([6,5])
        # node repeatablity rates for one model
        node_repeatablity_rates = np.zeros([6,5])
        for repeat_time in np.arange(args.repeat_times):
            print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time+1))
            print("********** BUILD GCN NETWORK: GCN_{}Layers **********".format(gcn_model_layer))
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

            print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time+1))
            print("********** TRAIN GCN NETWORK: {} Layers **********".format(gcn_model_layer))
            if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed':
                train_citation(gcn, g, features, labels, train_mask, test_mask, args)
            elif args.dataset == 'ppi':
                train_ppi(gcn, train_dataloader, valid_dataloader, args)

            checkpoint_path = '../checkpoints/gcn_' + str(gcn_model_layer) + 'layers_' + args.dataset + '.pkl'
            checkpoint_file = os.path.join(os.getcwd(), checkpoint_path)
            torch.save(gcn.state_dict(), checkpoint_file)

            print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time+1))
            print("********** BUILD REGRESSION MODEL TO RECOVER INPUT FROM EMBEDDING: {} Layers **********".format(gcn_model_layer))
            identifiable_nodes_1layer = set() # store nodes that can be identified after 1st layer
            for intermediate_layer in np.arange(1, gcn_model_layer+1):
                identifiable_nodes_current_layer = set()  # store nodes that can be identified after current layer
                print("********** INTERMEDIATE LAYER: {} **********".format(intermediate_layer))
                if args.dataset in 'cora, citeseer, pubmed':
                    embedding = gcn(g, features)[-intermediate_layer].clone().detach().to(device)
                    input_features = features.clone().detach().to(device)
                    reg_in = embedding.shape[1]
                    reg_out = input_features.shape[1]
                    reg_h = config['regression_hidden_features']
                elif args.dataset == 'ppi':
                    for data in train_dataset:
                        data[0].ndata['embedding'] = gcn(data[0], data[0].ndata['feat'].float(), args)[1].detach()
                    for data in valid_dataset:
                        data[0].ndata['embedding'] = gcn(data[0], data[0].ndata['feat'].float(), args)[1].detach()
                    reg_in = data[0].ndata['embedding'].shape[1]
                    reg_out = data[0].ndata['feat'].shape[1]
                    reg_h = config['regression_hidden_features']

                if args.regression_model == 'mlp':
                    reg = MLP(reg_in, reg_h, reg_out)
                elif args.regression_model == 'slp':
                    reg = SLP(reg_in, reg_out)
                elif args.regression_model == 'none':
                    pass

                print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time+1))
                print("********** TRAIN REGRESSION MODEL TO RECOVER INPUT FROM EMBEDDING: {} Layers **********".format(gcn_model_layer))
                print("********** INTERMEDIATE LAYER: {} **********".format(intermediate_layer))
                if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed':
                    train_reg_citation(reg, embedding, input_features, train_mask, test_mask, args)
                elif args.dataset == 'ppi':
                    train_reg_ppi(reg, train_dataloader, valid_dataloader, args)

                print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time+1))
                print("********** NEAREST NEIGHBOUR TO FIND CORRESPONDING INPUT: {} Layers **********".format(gcn_model_layer))
                print("********** INTERMEDIATE LAYER: {} **********".format(intermediate_layer))
                if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed':
                    nn_list = np.zeros(g.number_of_nodes())   # indices of the found nearest neighbourhood of nodes
                    if args.regression_model in 'mlp, slp':
                        reg_output = reg(embedding)
                    else:
                        reg_output = embedding.clone().detach().to(device)
                    for node_ind in range(g.number_of_nodes()):
                        neighbour_ind = (g.adjacency_matrix(transpose=True)[node_ind].to_dense()==1)
                        neighbour_feat = input_features[neighbour_ind]
                        node_reg_output = reg_output[node_ind]
                        # Find Nearest Neighbour
                        if args.regression_metric == 'l2':
                            dist = torch.norm(neighbour_feat - node_reg_output, dim=1, p=None)
                            nn = dist.topk(1, largest=False)
                        elif args.regression_metric == 'cos':
                            node_reg_output = node_reg_output.expand_as(neighbour_feat)
                            dist = torch.nn.functional.cosine_similarity(neighbour_feat, node_reg_output)
                            nn = dist.topk(1, largest=False)
                        # record the index of nn
                        nn_list[node_ind] = g.adjacency_matrix(transpose=True)[node_ind]._indices()[0,nn.indices].item()
                        print('Node_id: {} | Corresponding NN Node_id: {}'.format(node_ind, nn_list[node_ind]))

                print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time+1))
                print("********** COMPUTE IDENTIFIABILITY **********")
                print("********** INTERMEDIATE LAYER: {} **********".format(intermediate_layer))
                if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed':
                    num_identifiable = 0.0
                    num_repeatable = 0.0
                    for node_ind in range(g.number_of_nodes()):
                        if node_ind == nn_list[node_ind]: # whether identifiable
                            num_identifiable+=1
                            identifiable_nodes_current_layer.add(node_ind)
                            if intermediate_layer==1:
                                identifiable_nodes_1layer.add(node_ind)
                    identifiability_rate = num_identifiable / g.number_of_nodes()
                    for node_ind in identifiable_nodes_current_layer:
                        if node_ind in identifiable_nodes_1layer:
                            num_repeatable+=1
                    node_repeatability_rate = num_repeatable / len(identifiable_nodes_current_layer)
                print('identifiability_rate = {} %'.format(identifiability_rate*100))
                print('node_repeatability_rate = {} %'.format(node_repeatability_rate * 100))
                embedding_id_rates[intermediate_layer-1, repeat_time] = identifiability_rate
                node_repeatablity_rates[intermediate_layer-1, repeat_time] = node_repeatability_rate
        df_column_ind = models[gcn_model_layer-2]
        embedding_id_rates_df[df_column_ind] = np.mean(embedding_id_rates, axis=1)
        node_repeatability_rates_df[df_column_ind] = np.mean(node_repeatablity_rates, axis=1)

    save_subpath = '../outputs/embedding_id_rates_' + args.dataset + '.csv'
    save_path = os.path.join(os.getcwd(), save_subpath)
    embedding_id_rates_df.to_csv(save_path)
    save_subpath = '../outputs/node_repeatability_rates_' + args.dataset + '.csv'
    save_path = os.path.join(os.getcwd(), save_subpath)
    node_repeatability_rates_df.to_csv(save_path)


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Try to find fixpoint")

    parser.add_argument('--dataset', default='cora', help='choose dataset from: cora, pubmed, citeseer, ppi')
    parser.add_argument('--regression_model', default='mlp', help='choose model structure from: slp, mlp or none')
    parser.add_argument('--regression_metric', default='cos', help='choose metric for regression and nearest neighbour from: l2 or cos')
    parser.add_argument('--knn', type=int, default=1, help='method to find the corresponding node among neighboring nodes after recovery, k=1,2 or 3')
    parser.add_argument('--repeat_times', type=int, default=5, help='experiment repeating times for single layer')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")