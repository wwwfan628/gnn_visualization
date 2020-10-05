from src.utils.dataset import load_dataset
from src.utils.train_embedding_id import train_gcn, train_regression, classify_nodes
from src.models.gcn_identifiability import GCN
from src.models.mlp_embedding_id import MLP

import argparse
import torch
import os
import yaml
import pandas as pd
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("RUNNING ON: {}".format(device))


def main(args):

    # check if 'outputs' and 'outputs/identifiability' directory exist, if not create
    outputs_dir = os.path.join(os.getcwd(), '../outputs')
    outputs_subdir = os.path.join(outputs_dir, 'identifiability')
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    if not os.path.exists(outputs_subdir):
        os.makedirs(outputs_subdir)

    # load dataset
    print("********** LOAD DATASET **********")
    g, features, labels, train_mask, valid_mask, test_mask = load_dataset(args)

    # prepare to build network
    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    h_feats = config['hidden_features']
    in_feats = features.shape[1]
    out_feats = torch.max(labels).item() + 1

    # result before averaging over experiments'iterations
    # store 'identifiability rates'/'repeating rates' for differnt iterations/layers/models
    # 1st dim.:repeating time  2nd dim.:layer  3rd dim.:model
    identifiability_rates = np.zeros([args.repeat_times, args.max_gcn_layers, args.max_gcn_layers])
    repeating_rates = np.zeros([args.repeat_times, args.max_gcn_layers, args.max_gcn_layers])
    # store 'accracy'/'accuracy on identifiable nodes'/'accuracy on unidentifiable nodes' for different iterations/models
    # 1st dim.:repeating time  2nd dim.:model
    accuracy = np.zeros([args.repeat_times, args.max_gcn_layers])
    accuracy_id = np.zeros([args.repeat_times, args.max_gcn_layers])
    accuracy_unid = np.zeros([args.repeat_times, args.max_gcn_layers])

    for repeat_time in np.arange(args.repeat_times):
        for gcn_model_layer in np.arange(1, args.max_gcn_layers+1):
            print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time+1))
            print("********** GCN MODEL: GCN_{}layers **********".format(gcn_model_layer))
            print("********** BUILD GCN NETWORK**********")
            gcn = GCN(gcn_model_layer, in_feats, h_feats, out_feats).to(device)

            print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time+1))
            print("********** GCN MODEL: GCN_{}layer **********".format(gcn_model_layer))
            print("********** TRAIN GCN NETWORK **********")
            acc = train_gcn(gcn, g, features, labels, train_mask, valid_mask, args)
            accuracy[repeat_time, gcn_model_layer-1] = acc
            correct_classified_nodes, incorrect_classified_nodes = classify_nodes(gcn, g, features, labels)

            print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time + 1))
            print("********** GCN MODEL: GCN_{}layer **********".format(gcn_model_layer))
            identifiable_nodes_1layer = set()  # store nodes that can be identified after 1st layer
            identifiable_nodes_last_layer = set()  # store nodes that can be identified after last layer
            for intermediate_layer in np.arange(1, gcn_model_layer+1):
                identifiable_nodes_current_layer = set()  # store nodes that can be identified after current layer
                print("********** INTERMEDIATE LAYER: {} **********".format(intermediate_layer))
                print("********** BUILD REGRESSION MODEL TO RECOVER INPUT FROM EMBEDDING **********")
                # prepare to build the regression model
                embedding = gcn(g, features)[intermediate_layer].clone().detach().to(device)
                input_features = features.clone().detach().to(device)
                regression_in = embedding.shape[1]
                regression_out = input_features.shape[1]
                regression_h = config['regression_hidden_features_identity']

                regression_model = MLP(regression_in, regression_h, regression_out)   # regression model

                print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time + 1))
                print("********** GCN MODEL: GCN_{}layer **********".format(gcn_model_layer))
                print("********** INTERMEDIATE LAYER: {} **********".format(intermediate_layer))
                print("********** TRAIN REGRESSION MODEL TO RECOVER INPUT FROM EMBEDDING **********")
                train_regression(regression_model, embedding, input_features, train_mask, test_mask, args)

                print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time + 1))
                print("********** GCN MODEL: GCN_{}layer **********".format(gcn_model_layer))
                print("********** INTERMEDIATE LAYER: {} **********".format(intermediate_layer))
                print("********** K-NN FINDING CORRESPONDING INPUT FEATURES **********")
                num_nodes = g.number_of_nodes()
                nn_list = np.zeros([num_nodes, args.knn])   # indices of the found nearest neighbourhood of nodes
                regression_output = regression_model(embedding)
                for node_ind in range(num_nodes):
                    neighbour_ind = g.adjacency_matrix(transpose=True)[node_ind].to_dense() == 1
                    neighbour_feat = input_features[neighbour_ind]
                    node_regression_output = regression_output[node_ind]
                    # Find k Nearest Neighbourhood
                    node_regression_output = node_regression_output.expand_as(neighbour_feat)
                    dist = torch.nn.functional.cosine_similarity(neighbour_feat, node_regression_output)
                    nn = dist.topk(args.knn, largest=False)
                    # record the index of nn
                    nn_list[node_ind] = g.adjacency_matrix(transpose=True)[node_ind]._indices()[0, nn.indices]
                    print('Node_id: {} | Corresponding NN Node_id: {}'.format(node_ind, nn_list[node_ind]))

                print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time + 1))
                print("********** GCN MODEL: GCN_{}layer **********".format(gcn_model_layer))
                print("********** INTERMEDIATE LAYER: {} **********".format(intermediate_layer))
                print("********** COMPUTE IDENTIFIABILITY/REPEATING RATES **********")
                # compute number of identifiable nodes
                nodes_indices = np.arange(num_nodes)
                nodes_indices_expansion = np.expand_dims(nodes_indices, args.knn)
                identifiable_nodes_current_layer.update(nodes_indices[np.any(nodes_indices_expansion.repeat(args.knn,axis=1)==nn_list,axis=1)])
                num_identifiable = len(identifiable_nodes_current_layer)

                if intermediate_layer == 1:
                    identifiable_nodes_1layer.update(identifiable_nodes_current_layer)
                if intermediate_layer == gcn_model_layer:
                    # compute accuracy on identifiable nodes and unidentifiable nodes
                    identifiable_nodes_last_layer.update(identifiable_nodes_current_layer)
                    nodes = set(nodes_indices)
                    unidentifiable_nodes_last_layer = nodes.difference(identifiable_nodes_last_layer)
                    id_correct = len(identifiable_nodes_last_layer.intersection(correct_classified_nodes))
                    id_incorrect = len(identifiable_nodes_last_layer.intersection(incorrect_classified_nodes))
                    unid_correct = len(unidentifiable_nodes_last_layer.intersection(correct_classified_nodes))
                    unid_incorrect = len(unidentifiable_nodes_last_layer.intersection(incorrect_classified_nodes))
                    accuracy_id[repeat_time, gcn_model_layer - 1] = id_correct * 1.0/(id_correct + id_incorrect)
                    accuracy_unid[repeat_time, gcn_model_layer - 1] = unid_correct * 1.0/(unid_correct + unid_incorrect)
                    print('accuracy on identifiable nodes = {} %'.format(accuracy_id[repeat_time, gcn_model_layer - 1]*100))
                    print('accuracy on unidentifiable nodes = {} %'.format(accuracy_unid[repeat_time, gcn_model_layer - 1] * 100))

                # compute number of repeating nodes
                num_repeating = len(identifiable_nodes_current_layer.intersection(identifiable_nodes_1layer))

                identifiability_rate = num_identifiable * 1.0 / num_nodes
                if len(identifiable_nodes_1layer)!=0:
                    repeating_rate = num_repeating * 1.0 / len(identifiable_nodes_1layer)
                else:
                    repeating_rate = 0.0
                print('identifiability_rate = {} %'.format(identifiability_rate*100))
                print('node_repeatability_rate = {} %'.format(repeating_rate * 100))

                identifiability_rates[repeat_time, intermediate_layer-1, gcn_model_layer-1] = identifiability_rate
                repeating_rates[repeat_time, intermediate_layer-1, gcn_model_layer-1] = repeating_rate

    print("********** SAVE RESULT **********")
    # final result
    # dataframe to store 'identifiability rates'/'repeating rates' for different layers/models
    # row:layer  column:model
    models = []  # dataframe column index
    for gcn_model_layer in np.arange(1, args.max_gcn_layers + 1):
        model_name = str(gcn_model_layer) + '-layers ' + 'GCN'
        models.append(model_name)
    identifiability_rates_df = pd.DataFrame(np.mean(identifiability_rates, axis=0), index=np.arange(1, args.max_gcn_layers + 1), columns=models)
    repeating_rates_df = pd.DataFrame(np.mean(repeating_rates, axis=0), index=np.arange(1, args.max_gcn_layers + 1), columns=models)
    accuracy_df = pd.DataFrame(np.mean(accuracy, axis=0), index=models, columns=['accuracy'])
    accuracy_id_unid_array = np.array([np.mean(accuracy_id, axis=0),np.mean(accuracy_unid, axis=0)]).transpose()
    accuracy_id_unid_df = pd.DataFrame(accuracy_id_unid_array, index=models, columns=['accuracy on identifiable nodes', 'accuracy on unidentifiable nodes'])

    # save result in csv files
    save_subpath = 'embedding_id_rates_' + args.dataset + '.csv'
    save_path = os.path.join(outputs_subdir, save_subpath)
    identifiability_rates_df.to_csv(save_path)
    save_subpath = 'repeating_rates_' + args.dataset + '.csv'
    save_path = os.path.join(outputs_subdir, save_subpath)
    repeating_rates_df.to_csv(save_path)
    save_subpath = 'accuracy_' + args.dataset + '.csv'
    save_path = os.path.join(outputs_subdir, save_subpath)
    accuracy_df.to_csv(save_path)
    save_subpath = 'accuracy_id_unid_' + args.dataset + '.csv'
    save_path = os.path.join(outputs_subdir, save_subpath)
    accuracy_id_unid_df.to_csv(save_path)


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Identifiability")

    parser.add_argument('--dataset', default='cora', help='choose dataset from: cora, pubmed, citeseer')
    parser.add_argument('--knn', type=int, default=1, help='find k nearest neighbourhoods among input features after recovery')
    parser.add_argument('--repeat_times', type=int, default=10, help='experiment repeating times for single layer')
    parser.add_argument('--max_gcn_layers', type=int, default=10, help='the maxmal gcn models\'s layer')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")