from src.utils.dataset import load_dataset
from src.utils.train_embedding_id import train_citation, train_ppi
from src.models.gcn_embedding_id import GCN_Baseline, GCN_Baseline_3Layers

import argparse
import torch
import os
import numpy as np


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
    print("********** BUILD GCN NETWORK **********")
    if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed':
        h_feats = 16
    elif args.dataset == 'ppi':
        h_feats = 128

    if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed':
        in_feats = features.shape[1]
        out_feats = torch.max(labels).item() + 1
    elif args.dataset == 'ppi':
        in_feats = train_dataset.features.shape[1]
        out_feats = train_dataset.labels.shape[1]

    if args.gcn_model == '3':
        gcn_baseline = GCN_Baseline_3Layers(in_feats, h_feats, out_feats).to(device)
    elif args.gcn_model == '2':
        gcn_baseline = GCN_Baseline(in_feats, h_feats, out_feats).to(device)

    print("********** TRAIN GCN NETWORK **********")
    if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed':
        train_citation(gcn_baseline, g, features, labels, train_mask, test_mask, args)
    elif args.dataset == 'ppi':
        train_ppi(gcn_baseline, train_dataloader, valid_dataloader, args)

    checkpoint_path = '../checkpoints/slp_gcn_' + args.dataset + '.pkl'
    checkpoint_file = os.path.join(os.getcwd(), checkpoint_path)
    torch.save(gcn_baseline.state_dict(), checkpoint_file)

    # print("********** COMPUTE GRADIENTS **********")
    # if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed':
    #     inputs = features.clone().detach().requires_grad_(True).to(device)
    #     gcn_baseline.eval()
    #     embedding = gcn_baseline(g,inputs,args)[1]
    #     inputs_gradient = np.zeros([embedding.shape[0],embedding.shape[1],inputs.shape[0],inputs.shape[1]])
    #     for row_id in range(embedding.shape[0]):
    #         for col_id in range(embedding.shape[1]):
    #             print('row: {} | col: {}'.format(row_id, col_id))
    #             # compute gradient for embedding[row_id,col_id]
    #             gradients = torch.zeros(embedding.shape).float()
    #             gradients[row_id,col_id] = 1.0
    #             embedding.backward(gradient=gradients,retain_graph=True)
    #             inputs_gradient[row_id,col_id,:,:] = inputs.grad
    #             inputs.grad.data.zero_() # set gradient to 0

    print("********** COMPUTE CONTRIBUTION **********")
    if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed':
        inputs = features.clone().detach().requires_grad_(True).to(device)
        gcn_baseline.eval()
        embedding = gcn_baseline(g,inputs,args)[1]
        inputs_gradient = np.zeros([embedding.shape[1], inputs.shape[0], inputs.shape[1]])
        for embedding_node_id in range(embedding.shape[0]):
            # compute gradient for embedding[embedding_node_id,:] wrt inputs[input_node_id,:]
            for embedding_col_id in range(embedding.shape[1]):
                gradients = torch.zeros(embedding.shape).float()
                gradients[embedding_node_id,embedding_col_id] = 1.0
                embedding.backward(gradient=gradients,retain_graph=True)
                inputs_gradient[embedding_col_id,:,:] = inputs.grad
                inputs.grad.data.zero_() # set gradient to 0
            inputs_contribution_denormalized = np.linalg.norm(inputs_gradient,axis=(0,2),ord='fro')
            max_contribution_input_node = np.argmax(inputs_contribution_denormalized)
            self_contribution_normalized = inputs_contribution_denormalized[embedding_node_id] / np.sum(inputs_contribution_denormalized)
            max_contribution_normalized = inputs_contribution_denormalized[max_contribution_input_node] / np.sum(inputs_contribution_denormalized)
            print('Embedding node id {} | Self Contribution {} | Max Cotributed Input Node {} | Max Contribution {}'.format(embedding_node_id, self_contribution_normalized, max_contribution_input_node, max_contribution_normalized))


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Try to find fixpoint")

    parser.add_argument('dataset', help='choose dataset from: cora, reddit-self-loop, ppi, aids, reddit-binary and imdb-binary')
    parser.add_argument('gcn_model', help='choose GCN model layers from: 2 or 3')
    parser.add_argument('embedding_layer', help='From which layer do you want to check the embedding\'s identifiability?')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")