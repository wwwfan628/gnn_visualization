from src.utils.dataset import load_dataset
from src.utils.train_embedding_id import train_citation, classify_nodes_citation
from src.models.gcn_embedding_id import GCN_1Layer,GCN_2Layers,GCN_3Layers,GCN_4Layers,GCN_5Layers,GCN_6Layers,GCN_7Layers,GCN_8Layers

import argparse
import torch
import os
import numpy as np
import yaml
import dgl
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("RUNNING ON: {}".format(device))

def main(args):

    # check if 'outputs' and 'checkpoints' directory exist, if not create
    outputs_dir = os.path.join(os.getcwd(), '../outputs')
    outputs_subdir = os.path.join(outputs_dir, args.save_dir_name)
    checkpoints_dir = os.path.join(os.getcwd(), '../checkpoints')
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    if not os.path.exists(outputs_subdir):
        os.makedirs(outputs_subdir)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    if args.dataset in 'cora, pubmed, citeseer':
        # load dataset
        print("********** LOAD DATASET **********")
        g, features, labels, train_mask, valid_mask, test_mask = load_dataset(args)

        accuracy = np.zeros([args.repeat_time, args.max_gcn_layer]) # store accuracy for different models
        # store contribution
        # 1st dim.:repeat time    2nd dim.:models    3rd dim.:n-hop neighbourhoods
        contribution = np.zeros([args.repeat_time, args.max_gcn_layer, args.max_gcn_layer+1])

        # prepare to build network
        path = '../configs/' + args.dataset + '.yaml'
        config_file = os.path.join(os.getcwd(), path)
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        h_feats = config['hidden_features']
        in_feats = features.shape[1]
        out_feats = torch.max(labels).item() + 1

        for repeat_time in np.arange(args.repeat_time):
            for gcn_layer in range(1, args.max_gcn_layer+1):
                # build network
                print("********** GCN MODEL: GCN_{}layer **********".format(gcn_layer))
                print("********** BUILD GCN NETWORK **********")
                if gcn_layer == 1:
                    gcn = GCN_1Layer(in_feats, out_feats).to(device)
                elif gcn_layer == 2:
                    gcn = GCN_2Layers(in_feats, h_feats, out_feats).to(device)
                elif gcn_layer == 3:
                    gcn = GCN_3Layers(in_feats, h_feats, out_feats).to(device)
                elif gcn_layer == 4:
                    gcn = GCN_4Layers(in_feats, h_feats, out_feats).to(device)
                elif gcn_layer == 5:
                    gcn = GCN_5Layers(in_feats, h_feats, out_feats).to(device)
                elif gcn_layer == 6:
                    gcn = GCN_6Layers(in_feats, h_feats, out_feats).to(device)
                elif gcn_layer == 7:
                    gcn = GCN_7Layers(in_feats, h_feats, out_feats).to(device)
                elif gcn_layer == 8:
                    gcn = GCN_8Layers(in_feats, h_feats, out_feats).to(device)

                print("********** GCN MODEL: GCN_{}layer **********".format(gcn_layer))
                print("********** TRAIN GCN NETWORK **********")
                acc = train_citation(gcn, g, features, labels, train_mask, test_mask, args)
                accuracy[repeat_time, gcn_layer-1] = acc
                correct_classified_nodes, incorrect_classified_nodes = classify_nodes_citation(gcn, g, features, labels)

                checkpoint_path = str(gcn_layer) + '_layers_gcn_' + str(repeat_time) + '_' + args.dataset  + '.pkl'
                checkpoint_file = os.path.join(checkpoints_dir, checkpoint_path)
                torch.save(gcn.state_dict(), checkpoint_file)

                print("********** GCN MODEL: GCN_{}layer **********".format(gcn_layer))
                print("********** COMPUTE CONTRIBUTION **********")
                # store contribution of n-hop neighbourhoods
                # 1st dim.:node id    2nd dim.:n-hop neighbours
                contribution_hop = np.zeros([g.number_of_nodes(),args.max_gcn_layer + 1])
                inputs = features.clone().detach().requires_grad_(True).to(device)
                gcn.eval()
                embedding_last_layer = gcn(g,inputs)[0]
                inputs_gradient = np.zeros([embedding_last_layer.shape[1], inputs.shape[0], inputs.shape[1]])
                for node_id in range(embedding_last_layer.shape[0]):
                    print('Node ID {} '.format(node_id))
                    # compute gradient for embedding[embedding_node_id,:] wrt inputs[input_node_id,:]
                    for embedding_col_id in range(embedding_last_layer.shape[1]):
                        gradients = torch.zeros(embedding_last_layer.shape).float().to(device)
                        gradients[node_id,embedding_col_id] = 1.0
                        embedding_last_layer.backward(gradient=gradients,retain_graph=True)
                        inputs_gradient[embedding_col_id,:,:] = inputs.grad
                        inputs.grad.data.zero_() # set gradient to 0
                    inputs_contribution_denormalized = np.linalg.norm(inputs_gradient,axis=(0,2),ord='fro')
                    inputs_contribution_normalized = inputs_contribution_denormalized / np.sum(inputs_contribution_denormalized)

                    # BFS find n-hop neighbourhoods
                    neighbour_list = list(dgl.bfs_nodes_generator(g, node_id))
                    if len(neighbour_list)<=args.max_gcn_layer:
                        for hop_ind in np.arange(len(neighbour_list)):
                            contribution_hop[node_id, hop_ind] = np.sum(inputs_contribution_normalized[neighbour_list[hop_ind]])
                    else:
                        for hop_ind in np.arange(args.max_gcn_layer+1):
                            contribution_hop[node_id, hop_ind] = np.sum(inputs_contribution_normalized[neighbour_list[hop_ind]])

                contribution[repeat_time, gcn_layer-1, :] = np.sum(contribution_hop/np.sum(contribution_hop), axis=0)
                print('Current Model Contribution: {} '.format(contribution[repeat_time, gcn_layer-1, :]))

        print("********** SAVE RESULT **********")
        models = []  # dataframe index
        for gcn_layer in np.arange(1, args.max_gcn_layer + 1):
            if gcn_layer == 1:
                model_name = 'gcn_1layer'
            else:
                model_name = 'gcn_' + str(gcn_layer) + 'layers'
            models.append(model_name)

        hop = [] # dataframe index
        for hop_ind in np.arange(args.max_gcn_layer+1):
            neighbour_name = str(hop_ind)+'-hop'
            hop.append(neighbour_name)

        accuracy_df = pd.DataFrame(np.mean(accuracy,axis=0), index=models, columns=['accuracy'])
        contribution_df = pd.DataFrame(np.mean(contribution,axis=0), index=models, columns=hop)

        # save result in csv files
        save_subpath = 'accuracy_' + args.dataset + '.csv'
        save_path = os.path.join(outputs_subdir, save_subpath)
        accuracy_df.to_csv(save_path)
        save_subpath = 'contribution_' + args.dataset + '.csv'
        save_path = os.path.join(outputs_subdir, save_subpath)
        contribution_df.to_csv(save_path)


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Try to find fixpoint")

    parser.add_argument('--dataset', default='cora', help='choose dataset from: cora, reddit-self-loop, ppi, aids, reddit-binary and imdb-binary')
    parser.add_argument('--max_gcn_layer', type=int, default=8, help='choose max GCN model layers smaller than 10')
    parser.add_argument('--repeat_time', type=int, default=1, help='repeating time of experiments')
    parser.add_argument('--save_dir_name', default='contribution', help='saving directory\'s name')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")