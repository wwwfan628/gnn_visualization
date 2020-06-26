from src.utils.dataset import load_dataset
from src.utils.train_embedding_id import train_citation, train_ppi, train_regression_citation,train_regression_ppi, classify_nodes_citation
from src.models.gcn_embedding_id import GCN_1Layer,GCN_2Layers,GCN_3Layers,GCN_4Layers,GCN_5Layers,GCN_6Layers,GCN_7Layers,GCN_8Layers
from src.models.regression_embedding_id import MLP, SLP

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

        # prepare to build network
        path = '../configs/' + args.dataset + '.yaml'
        config_file = os.path.join(os.getcwd(), path)
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        h_feats = config['hidden_features']
        in_feats = features.shape[1]
        out_feats = torch.max(labels).item() + 1

        # result before averaging over experiments'iterations
        # store 'identifiability rates'/'repeating rates'/'1-hop neighbours rates' for differnt iterations/layers/models
        # 1st dim.:repeating time  2nd dim.:layer  3rd dim.:model
        identifiability_rates = np.zeros([args.repeat_times, args.max_gcn_layers, args.max_gcn_layers])
        repeating_rates = np.zeros([args.repeat_times, args.max_gcn_layers, args.max_gcn_layers])
        # store 'accracy'/'accuracy & identifiability' for different iterations/models
        # 1st dim.:repeating time  2nd dim.:model
        accuracy = np.zeros([args.repeat_times, args.max_gcn_layers])
        # 1st dim.:repeating time  2nd dim.:model  3rd dim.:identifiability-accuracy ++ +- -+ --
        relationship_id_acc = np.zeros([args.repeat_times, args.max_gcn_layers, 4])

        for repeat_time in np.arange(args.repeat_times):
            for gcn_model_layer in np.arange(1, args.max_gcn_layers+1):
                print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time+1))
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

                print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time+1))
                print("********** GCN MODEL: GCN_{}layer **********".format(gcn_model_layer))
                print("********** TRAIN GCN NETWORK **********")
                acc = train_citation(gcn, g, features, labels, train_mask, test_mask, args)
                accuracy[repeat_time, gcn_model_layer-1] = acc
                correct_classified_nodes, incorrect_classified_nodes = classify_nodes_citation(gcn, g, features, labels)

                checkpoint_file_name = 'gcn_' + str(gcn_model_layer) + 'layers_' + args.dataset + '.pkl'
                checkpoint_file = os.path.join(checkpoints_dir, checkpoint_file_name)
                torch.save(gcn.state_dict(), checkpoint_file)

                print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time + 1))
                print("********** GCN MODEL: GCN_{}layer **********".format(gcn_model_layer))
                identifiable_nodes_1layer = set()  # store nodes that can be identified after 1st layer
                identifiable_nodes_last_layer = set()  # store nodes that can be identified after last layer
                for intermediate_layer in np.arange(1, gcn_model_layer+1):
                    identifiable_nodes_current_layer = set()  # store nodes that can be identified after current layer
                    print("********** INTERMEDIATE LAYER: {} **********".format(intermediate_layer))
                    print("********** BUILD REGRESSION MODEL TO RECOVER INPUT FROM EMBEDDING **********")
                    # prepare to build the regression model
                    embedding = gcn(g, features)[-intermediate_layer].clone().detach().to(device)
                    input_features = features.clone().detach().to(device)
                    regression_in = embedding.shape[1]
                    regression_out = input_features.shape[1]
                    regression_h = config['regression_hidden_features_identity']

                    if args.regression_model == 'mlp':
                        regression_model = MLP(regression_in, regression_h, regression_out)
                    elif args.regression_model == 'slp':
                        regression_model = SLP(regression_in, regression_out)

                    print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time + 1))
                    print("********** GCN MODEL: GCN_{}layer **********".format(gcn_model_layer))
                    print("********** INTERMEDIATE LAYER: {} **********".format(intermediate_layer))
                    print("********** TRAIN REGRESSION MODEL TO RECOVER INPUT FROM EMBEDDING **********")
                    train_regression_citation(regression_model, embedding, input_features, train_mask, test_mask, args)

                    print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time + 1))
                    print("********** GCN MODEL: GCN_{}layer **********".format(gcn_model_layer))
                    print("********** INTERMEDIATE LAYER: {} **********".format(intermediate_layer))
                    print("********** K-NN FINDING CORRESPONDING INPUT FEATURES ON TEST SET **********")
                    test_set_nodes_indices = np.arange(g.number_of_nodes())[test_mask]
                    num_test_nodes = len(test_set_nodes_indices)
                    nn_list = np.zeros([num_test_nodes, args.knn])   # indices of the found nearest neighbourhood of nodes
                    regression_output = regression_model(embedding)
                    for ind, test_node_ind in enumerate(test_set_nodes_indices):
                        neighbour_ind = g.adjacency_matrix(transpose=True)[test_node_ind].to_dense() == 1
                        neighbour_feat = input_features[neighbour_ind]
                        node_regression_output = regression_output[test_node_ind]
                        # Find k Nearest Neighbourhood
                        if args.regression_metric == 'l2':
                            dist = torch.norm(neighbour_feat - node_regression_output, dim=1, p=None)
                            nn = dist.topk(args.knn, largest=False)
                        elif args.regression_metric == 'cos':
                            node_regression_output = node_regression_output.expand_as(neighbour_feat)
                            dist = torch.nn.functional.cosine_similarity(neighbour_feat, node_regression_output)
                            nn = dist.topk(args.knn, largest=False)
                        # record the index of nn
                        nn_list[ind] = g.adjacency_matrix(transpose=True)[test_node_ind]._indices()[0, nn.indices]
                        print('Node_id: {} | Corresponding NN Node_id: {}'.format(test_node_ind, nn_list[ind]))

                    print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time + 1))
                    print("********** GCN MODEL: GCN_{}layer **********".format(gcn_model_layer))
                    print("********** INTERMEDIATE LAYER: {} **********".format(intermediate_layer))
                    print("********** COMPUTE IDENTIFIABILITY/REPEATING RATES ON TEST SET **********")
                    # compute number of identifiable nodes
                    test_set_nodes_indices_expansion = np.expand_dims(test_set_nodes_indices,1)
                    identifiable_nodes_current_layer.update(test_set_nodes_indices[np.any(test_set_nodes_indices_expansion.repeat(args.knn,axis=1)==nn_list,axis=1)])
                    num_identifiable = len(identifiable_nodes_current_layer)

                    if intermediate_layer == 1:
                        identifiable_nodes_1layer.update(identifiable_nodes_current_layer)
                    if intermediate_layer == gcn_model_layer:
                        # compute relationship between accuracy & identifiability
                        identifiable_nodes_last_layer.update(identifiable_nodes_current_layer)
                        test_nodes = set(test_set_nodes_indices)
                        unidentifiable_nodes_last_layer = test_nodes.difference(identifiable_nodes_last_layer)
                        positive_positive = len(identifiable_nodes_last_layer.intersection(correct_classified_nodes))*1.0/num_test_nodes
                        positive_negative = len(identifiable_nodes_last_layer.intersection(incorrect_classified_nodes))*1.0/num_test_nodes
                        negative_positive = len(unidentifiable_nodes_last_layer.intersection(correct_classified_nodes))*1.0/num_test_nodes
                        negative_negative = len(unidentifiable_nodes_last_layer.intersection(incorrect_classified_nodes))*1.0/num_test_nodes
                        print('++ = {} %'.format(positive_positive * 100))
                        print('+- = {} %'.format(positive_negative * 100))
                        print('-+ = {} %'.format(negative_positive * 100))
                        print('-- = {} %'.format(negative_negative * 100))
                        relationship_id_acc[repeat_time, gcn_model_layer - 1, :] = np.array([positive_positive, positive_negative, negative_positive, negative_negative])

                    # compute number of repeating nodes
                    num_repeating = len(identifiable_nodes_current_layer.intersection(identifiable_nodes_1layer))

                    identifiability_rate = num_identifiable * 1.0 / num_test_nodes
                    if len(identifiable_nodes_current_layer)!=0:
                        repeating_rate = num_repeating * 1.0 / len(identifiable_nodes_current_layer)
                    else:
                        repeating_rate = 0.0
                    print('identifiability_rate = {} %'.format(identifiability_rate*100))
                    print('node_repeatability_rate = {} %'.format(repeating_rate * 100))

                    identifiability_rates[repeat_time, intermediate_layer-1, gcn_model_layer-1] = identifiability_rate
                    repeating_rates[repeat_time, intermediate_layer-1, gcn_model_layer-1] = repeating_rate

        print("********** SAVE RESULT **********")
        # final result
        # dataframe to store 'identifiability rates'/'repeating rates'/'1-hop neighbours rates' for different layers/models
        # row:layer  column:model
        models = []  # dataframe column index
        for gcn_model_layer in np.arange(1, args.max_gcn_layers + 1):
            if gcn_model_layer == 1:
                model_name = 'gcn_1layer'
            else:
                model_name = 'gcn_' + str(gcn_model_layer) + 'layers'
            models.append(model_name)
        identifiability_rates_df = pd.DataFrame(np.mean(identifiability_rates, axis=0), index=np.arange(1, args.max_gcn_layers + 1), columns=models)
        repeating_rates_df = pd.DataFrame(np.mean(repeating_rates, axis=0), index=np.arange(1, args.max_gcn_layers + 1), columns=models)
        accuracy_df = pd.DataFrame(np.mean(accuracy, axis=0), index=models, columns=['accuracy'])
        relationship_id_acc_df = pd.DataFrame(np.mean(relationship_id_acc, axis=0), index=models, columns=['++', '+-', '-+', '--'])

        # save result in csv files
        save_subpath = 'embedding_id_rates_' + args.dataset + '_'+ str(args.knn) + '-nn' + '.csv'
        save_path = os.path.join(outputs_subdir, save_subpath)
        identifiability_rates_df.to_csv(save_path)
        save_subpath = 'repeating_rates_' + args.dataset + '_' + str(args.knn) +'-nn' + '.csv'
        save_path = os.path.join(outputs_subdir, save_subpath)
        repeating_rates_df.to_csv(save_path)
        save_subpath = 'accuracy_' + args.dataset + '_' + str(args.knn) + '-nn' + '.csv'
        save_path = os.path.join(outputs_subdir, save_subpath)
        accuracy_df.to_csv(save_path)
        save_subpath = 'relatiship_id_accuracy_' + args.dataset + '_' + str(args.knn) + '-nn' + '.csv'
        save_path = os.path.join(outputs_subdir, save_subpath)
        relationship_id_acc_df.to_csv(save_path)


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Try to find fixpoint")

    parser.add_argument('--dataset', default='cora', help='choose dataset from: cora, pubmed, citeseer, ppi')
    parser.add_argument('--info', default='self-identity', help='choose the information to recover')
    parser.add_argument('--regression_model', default='mlp', help='choose model structure from: slp, mlp')
    parser.add_argument('--regression_metric', default='cos', help='choose metric for regression and nearest neighbour from: l2 or cos')
    parser.add_argument('--knn', type=int, default=1, help='method to find the corresponding node among neighboring nodes after recovery, k=1,2 or 3')
    parser.add_argument('--repeat_times', type=int, default=5, help='experiment repeating times for single layer')
    parser.add_argument('--max_gcn_layers', type=int, default=6, help='the maxmal gcn models\'s layer, not larger than 8')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")