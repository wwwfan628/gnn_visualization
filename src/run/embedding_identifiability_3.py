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
        neighbours_rates = np.zeros([args.repeat_times, args.max_gcn_layers, args.max_gcn_layers])
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
                acc = train_citation(gcn, g, features, labels, train_mask, valid_mask, args)
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
                    print("********** K-NN FINDING CORRESPONDING INPUT FEATURES **********")
                    num_nodes = g.number_of_nodes()
                    nn_list = np.zeros([num_nodes, args.knn])   # indices of the found nearest neighbourhood of nodes
                    regression_output = regression_model(embedding)
                    for node_ind in range(num_nodes):
                        neighbour_ind = g.adjacency_matrix(transpose=True)[node_ind].to_dense() == 1
                        neighbour_feat = input_features[neighbour_ind]
                        node_regression_output = regression_output[node_ind]
                        # Find k Nearest Neighbourhood
                        if args.regression_metric == 'l2':
                            dist = torch.norm(neighbour_feat - node_regression_output, dim=1, p=None)
                            nn = dist.topk(args.knn, largest=False)
                        elif args.regression_metric == 'cos':
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
                    nodes_indices_expansion = np.expand_dims(nodes_indices,1)
                    identifiable_nodes_current_layer.update(nodes_indices[np.any(nodes_indices_expansion.repeat(args.knn,axis=1)==nn_list,axis=1)])
                    num_identifiable = len(identifiable_nodes_current_layer)

                    if intermediate_layer == 1:
                        identifiable_nodes_1layer.update(identifiable_nodes_current_layer)
                    if intermediate_layer == gcn_model_layer:
                        # compute relationship between accuracy & identifiability
                        identifiable_nodes_last_layer.update(identifiable_nodes_current_layer)
                        nodes = set(nodes_indices)
                        unidentifiable_nodes_last_layer = nodes.difference(identifiable_nodes_last_layer)
                        positive_positive = len(identifiable_nodes_last_layer.intersection(correct_classified_nodes))*1.0/num_nodes
                        positive_negative = len(identifiable_nodes_last_layer.intersection(incorrect_classified_nodes))*1.0/num_nodes
                        negative_positive = len(unidentifiable_nodes_last_layer.intersection(correct_classified_nodes))*1.0/num_nodes
                        negative_negative = len(unidentifiable_nodes_last_layer.intersection(incorrect_classified_nodes))*1.0/num_nodes
                        print('++ = {} %'.format(positive_positive * 100))
                        print('+- = {} %'.format(positive_negative * 100))
                        print('-+ = {} %'.format(negative_positive * 100))
                        print('-- = {} %'.format(negative_negative * 100))
                        relationship_id_acc[repeat_time, gcn_model_layer - 1, :] = np.array([positive_positive, positive_negative, negative_positive, negative_negative])

                    # compute number of repeating nodes
                    num_repeating = len(identifiable_nodes_current_layer.intersection(identifiable_nodes_1layer))

                    # compute number of nodes that nn is its'neighbours
                    sparse_tensor_indices_row = np.expand_dims(np.arange(num_nodes),1).repeat(args.knn,axis=1).flatten()
                    sparse_tensor_indices_col = nn_list.flatten()
                    sparse_tensor_indices = torch.tensor([sparse_tensor_indices_row,sparse_tensor_indices_col],dtype=int)
                    sparse_tensor_values = torch.ones(len(sparse_tensor_indices_row),dtype=int)
                    nn_matrix = torch.sparse.LongTensor(sparse_tensor_indices,sparse_tensor_values,torch.Size([num_nodes,num_nodes])).to_dense().bool()
                    num_neighbours = len(np.arange(num_nodes)[(nn_matrix & g.adjacency_matrix(transpose=True).to_dense().bool()).any(1)])
                    num_neighbours-=num_identifiable  # graph contains self-loop

                    identifiability_rate = num_identifiable * 1.0 / num_nodes
                    if len(identifiable_nodes_current_layer)!=0:
                        repeating_rate = num_repeating * 1.0 / len(identifiable_nodes_current_layer)
                    else:
                        repeating_rate = 0.0
                    neighbours_rate = num_neighbours * 1.0 / num_nodes
                    print('identifiability_rate = {} %'.format(identifiability_rate*100))
                    print('node_repeatability_rate = {} %'.format(repeating_rate * 100))
                    print('neighbours_rate = {} %'.format(neighbours_rate * 100))

                    identifiability_rates[repeat_time, intermediate_layer-1, gcn_model_layer-1] = identifiability_rate
                    repeating_rates[repeat_time, intermediate_layer-1, gcn_model_layer-1] = repeating_rate
                    neighbours_rates[repeat_time, intermediate_layer-1, gcn_model_layer-1] = neighbours_rate

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
        neighbours_rates_df = pd.DataFrame(np.mean(neighbours_rates, axis=0), index=np.arange(1, args.max_gcn_layers + 1), columns=models)
        accuracy_df = pd.DataFrame(np.mean(accuracy, axis=0), index=models, columns=['accuracy'])
        relationship_id_acc_df = pd.DataFrame(np.mean(relationship_id_acc, axis=0), index=models, columns=['++', '+-', '-+', '--'])

        # save result in csv files
        save_subpath = 'embedding_id_rates_' + args.dataset + '_'+ str(args.knn) + '-nn' + '.csv'
        save_path = os.path.join(outputs_subdir, save_subpath)
        identifiability_rates_df.to_csv(save_path)
        save_subpath = 'repeating_rates_' + args.dataset + '_' + str(args.knn) +'-nn' + '.csv'
        save_path = os.path.join(outputs_subdir, save_subpath)
        repeating_rates_df.to_csv(save_path)
        save_subpath = 'neighbours_rates_' + args.dataset + '_' + str(args.knn) +'-nn' + '.csv'
        save_path = os.path.join(outputs_subdir, save_subpath)
        neighbours_rates_df.to_csv(save_path)
        save_subpath = 'accuracy_' + args.dataset + '_' + str(args.knn) + '-nn' + '.csv'
        save_path = os.path.join(outputs_subdir, save_subpath)
        accuracy_df.to_csv(save_path)
        save_subpath = 'relatiship_id_accuracy_' + args.dataset + '_' + str(args.knn) + '-nn' + '.csv'
        save_path = os.path.join(outputs_subdir, save_subpath)
        relationship_id_acc_df.to_csv(save_path)

    elif args.dataset == 'ppi':
        # load dataset
        print("********** LOAD DATASET **********")
        train_dataset, valid_dataset, test_dataset, train_dataloader, valid_dataloader, test_dataloader = load_dataset(args)

        # prepare to build network
        path = '../configs/' + args.dataset + '.yaml'
        config_file = os.path.join(os.getcwd(), path)
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        h_feats = config['hidden_features']
        in_feats = train_dataset.features.shape[1]
        out_feats = train_dataset.labels.shape[1]

        # result before averaging over experiments'iterations
        # store 'identifiability rates'/'repeating rates'/'1-hop neighbours rates' for differnt iterations/graphs/layers/models
        # 1st dim.:repeating time   2nd dim.:layer  3rd dim.:model   4th dim.:graph_id
        identifiability_rates = np.zeros([args.repeat_times, args.max_gcn_layers, args.max_gcn_layers, 24])
        repeating_rates = np.zeros([args.repeat_times, args.max_gcn_layers, args.max_gcn_layers, 24])
        neighbours_rates = np.zeros([args.repeat_times, args.max_gcn_layers, args.max_gcn_layers, 24])
        # store 'f1 score'/'f1 score & identifiability' for different iterations/models
        # 1st dim.:repeating time   2nd dim.:model
        f1_scores = np.zeros([args.repeat_times, args.max_gcn_layers])

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
                f1_score = train_ppi(gcn, train_dataloader, valid_dataloader)
                f1_scores[repeat_time, gcn_model_layer - 1] = f1_score

                checkpoint_file_name = 'gcn_' + str(gcn_model_layer) + 'layers_' + args.dataset + '.pkl'
                checkpoint_file = os.path.join(checkpoints_dir, checkpoint_file_name)
                torch.save(gcn.state_dict(), checkpoint_file)

                print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time + 1))
                print("********** GCN MODEL: GCN_{}layer **********".format(gcn_model_layer))
                identifiable_nodes_1layer = []   # store nodes that can be identified after 1st layer
                identifiable_nodes_last_layer = []   # store nodes that can be identified after last layer
                for intermediate_layer in np.arange(1, gcn_model_layer + 1):
                    print("********** INTERMEDIATE LAYER: {} **********".format(intermediate_layer))
                    print("********** BUILD REGRESSION MODEL TO RECOVER INPUT FROM EMBEDDING **********")
                    # prepare to build the regression model
                    for data in train_dataset:
                        data[0].ndata['embedding'] = gcn(data[0], data[0].ndata['feat'].float().to(device))[-intermediate_layer].clone().detach().to(device)
                    for data in valid_dataset:
                        data[0].ndata['embedding'] = gcn(data[0], data[0].ndata['feat'].float().to(device))[-intermediate_layer].clone().detach().to(device)
                    for data in test_dataset:
                        data[0].ndata['embedding'] = gcn(data[0], data[0].ndata['feat'].float().to(device))[-intermediate_layer].clone().detach().to(device)
                    regression_in = data[0].ndata['embedding'].shape[1]
                    regression_out = data[0].ndata['feat'].shape[1]
                    regression_h = config['regression_hidden_features_identity']

                    if args.regression_model == 'mlp':
                        regression_model = MLP(regression_in, regression_h, regression_out).to(device)
                    elif args.regression_model == 'slp':
                        regression_model = SLP(regression_in, regression_out).to(device)

                    print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time + 1))
                    print("********** GCN MODEL: GCN_{}layer **********".format(gcn_model_layer))
                    print("********** INTERMEDIATE LAYER: {} **********".format(intermediate_layer))
                    print("********** TRAIN REGRESSION MODEL TO RECOVER INPUT FROM EMBEDDING **********")
                    train_regression_ppi(regression_model, train_dataloader, valid_dataloader, args)

                    print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time + 1))
                    print("********** GCN MODEL: GCN_{}layer **********".format(gcn_model_layer))
                    print("********** INTERMEDIATE LAYER: {} **********".format(intermediate_layer))
                    ppi_dataset = [train_dataset, valid_dataset, test_dataset]
                    for dataset_id in np.arange(3):
                        for graph_id, data in enumerate(ppi_dataset[dataset_id]):
                            # compute graph_id wrt whole dataset
                            for ind in np.arange(dataset_id):
                                graph_id = graph_id + len(ppi_dataset[ind])
                            print("********** GRAPH ID: {} **********".format(graph_id+1))
                            print("********** K-NN FINDING CORRESPONDING INPUT FEATURES **********")
                            graph = data[0]
                            num_nodes = graph.number_of_nodes()
                            nn_list = np.zeros([num_nodes, args.knn])  # indices of the found nearest neighbourhood of nodes
                            regression_output = regression_model(graph.ndata['embedding'])
                            for node_ind in range(num_nodes):
                                neighbour_ind = graph.adjacency_matrix(transpose=True)[node_ind].to_dense() == 1
                                neighbour_feat = graph.ndata['feat'][neighbour_ind]
                                node_regression_output = regression_output[node_ind]
                                # Find k Nearest Neighbourhood
                                if args.regression_metric == 'l2':
                                    dist = torch.norm(neighbour_feat.to(device) - node_regression_output.to(device), dim=1, p=None)
                                    nn = dist.topk(args.knn, largest=False)
                                elif args.regression_metric == 'cos':
                                    node_regression_output = node_regression_output.expand_as(neighbour_feat)
                                    dist = torch.nn.functional.cosine_similarity(neighbour_feat.to(device), node_regression_output.to(device))
                                    nn = dist.topk(args.knn, largest=False)
                                # record the index of nn
                                nn_list[node_ind] = graph.adjacency_matrix(transpose=True)[node_ind]._indices()[0, nn.indices]
                                print('Node_id: {} | Corresponding NN Node_id: {}'.format(node_ind, nn_list[node_ind]))

                            print("********** EXPERIMENT ITERATION: {} **********".format(repeat_time + 1))
                            print("********** GCN MODEL: GCN_{}layer **********".format(gcn_model_layer))
                            print("********** INTERMEDIATE LAYER: {} **********".format(intermediate_layer))
                            print("********** GRAPH ID: {} **********".format(graph_id+1))
                            print("********** COMPUTE IDENTIFIABILITY/REPEATING RATES **********")
                            # compute number of identifiable nodes
                            nodes_indices = np.arange(num_nodes)
                            nodes_indices_expansion = np.expand_dims(nodes_indices, 1)
                            identifiable_nodes_current_layer = set(nodes_indices[np.any(nodes_indices_expansion.repeat(args.knn, axis=1) == nn_list, axis=1)])
                            num_identifiable = len(identifiable_nodes_current_layer)

                            if intermediate_layer == 1:
                                identifiable_nodes_1layer.append(identifiable_nodes_current_layer)
                            if intermediate_layer == gcn_model_layer:
                                identifiable_nodes_last_layer.append(identifiable_nodes_current_layer)

                            # compute number of repeating nodes
                            num_repeating = len(identifiable_nodes_current_layer.intersection(identifiable_nodes_1layer[graph_id]))

                            # compute number of nodes that nn is its'neighbours
                            sparse_tensor_indices_row = np.expand_dims(np.arange(num_nodes), 1).repeat(args.knn, axis=1).flatten()
                            sparse_tensor_indices_col = nn_list.flatten()
                            sparse_tensor_indices = torch.tensor([sparse_tensor_indices_row, sparse_tensor_indices_col],dtype=int)
                            sparse_tensor_values = torch.ones(len(sparse_tensor_indices_row), dtype=int)
                            nn_matrix = torch.sparse.LongTensor(sparse_tensor_indices, sparse_tensor_values,torch.Size([num_nodes, num_nodes])).to_dense().bool()
                            num_neighbours = len(np.arange(num_nodes)[(nn_matrix & graph.adjacency_matrix(transpose=True).to_dense().bool()).any(1)])
                            num_neighbours -= num_identifiable  # graph contains self-loop

                            identifiability_rate = num_identifiable * 1.0 / num_nodes
                            if len(identifiable_nodes_current_layer) != 0:
                                repeating_rate = num_repeating * 1.0 / len(identifiable_nodes_current_layer)
                            else:
                                repeating_rate = 0.0
                            neighbours_rate = num_neighbours * 1.0 / num_nodes
                            print('identifiability_rate = {} %'.format(identifiability_rate * 100))
                            print('node_repeatability_rate = {} %'.format(repeating_rate * 100))
                            print('neighbours_rate = {} %'.format(neighbours_rate * 100))

                            identifiability_rates[repeat_time, intermediate_layer - 1, gcn_model_layer - 1, graph_id] = identifiability_rate
                            repeating_rates[repeat_time, intermediate_layer - 1, gcn_model_layer - 1, graph_id] = repeating_rate
                            neighbours_rates[repeat_time, intermediate_layer - 1, gcn_model_layer - 1, graph_id] = neighbours_rate

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
        identifiability_rates_df = pd.DataFrame(np.mean(np.mean(identifiability_rates,axis=3), axis=0),index=np.arange(1, args.max_gcn_layers + 1), columns=models)
        repeating_rates_df = pd.DataFrame(np.mean(np.mean(repeating_rates,axis=3), axis=0), index=np.arange(1, args.max_gcn_layers + 1),columns=models)
        neighbours_rates_df = pd.DataFrame(np.mean(np.mean(neighbours_rates,axis=3), axis=0), index=np.arange(1, args.max_gcn_layers + 1), columns=models)
        f1_scores_df = pd.DataFrame(np.mean(f1_scores, axis=0), index=models, columns=['f1_score'])

        # save result in csv files
        save_subpath = 'embedding_id_rates_' + args.dataset + '_' + str(args.knn) + '-nn' + '.csv'
        save_path = os.path.join(outputs_subdir, save_subpath)
        identifiability_rates_df.to_csv(save_path)
        save_subpath = 'repeating_rates_' + args.dataset + '_' + str(args.knn) + '-nn' + '.csv'
        save_path = os.path.join(outputs_subdir, save_subpath)
        repeating_rates_df.to_csv(save_path)
        save_subpath = 'neighbours_rates_' + args.dataset + '_' + str(args.knn) + '-nn' + '.csv'
        save_path = os.path.join(outputs_subdir, save_subpath)
        neighbours_rates_df.to_csv(save_path)
        save_subpath = 'f1_scores_' + args.dataset + '_' + str(args.knn) + '-nn' + '.csv'
        save_path = os.path.join(outputs_subdir, save_subpath)
        f1_scores_df.to_csv(save_path)


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