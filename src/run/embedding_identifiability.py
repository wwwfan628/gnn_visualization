from src.utils.dataset import load_dataset
from src.utils.train_embedding_id import train_citation, train_ppi, train_reg_citation,train_reg_ppi
from src.models.gcn_embedding_id import GCN_Baseline, GCN_Baseline_3Layers
from src.models.regression_embedding_id import MLP, SLP

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

    print("********** BUILD REGRESSION MODEL TO RECOVER INPUT FROM EMBEDDING **********")
    if args.dataset in 'cora, citeseer, pubmed':
        embedding = gcn_baseline(g, features)[1].clone().detach().to(device)
        input_features = features.clone().detach().to(device)
        reg_in = embedding.shape[1]
        reg_out = input_features.shape[1]
        reg_h = 256
    elif args.dataset == 'ppi':
        for data in train_dataset:
            data[0].ndata['embedding'] = gcn_baseline(data[0], data[0].ndata['feat'].float())[1].detach()
        for data in valid_dataset:
            data[0].ndata['embedding'] = gcn_baseline(data[0], data[0].ndata['feat'].float())[1].detach()
        reg_in = data[0].ndata['embedding'].shape[1]
        reg_out = data[0].ndata['feat'].shape[1]
        reg_h = 64

    if args.regression_model == 'mlp':
        reg = MLP(reg_in, reg_h, reg_out)
    elif args.regression_model == 'slp':
        reg = SLP(reg_in, reg_out)
    elif args.regression_model == 'none':
        pass

    print("********** TRAIN REGRESSION MODEL TO RECOVER INPUT FROM EMBEDDING **********")
    if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed':
        train_reg_citation(reg, embedding, input_features, train_mask, test_mask, args)
    elif args.dataset == 'ppi':
        train_reg_ppi(reg, train_dataloader, valid_dataloader, args)

    print("********** NEAREST NEIGHBOUR TO FIND CORRESPONDING INPUT **********")
    if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed':
        nn_list = np.zeros(g.number_of_nodes())
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
                cos_dist = torch.nn.functional.cosine_similarity(dim=1)
                dist = cos_dist(neighbour_feat, node_reg_output)
                nn = dist.topk(1, largest=False)
            # record the index of nn
            nn_list[node_ind] = g.adjacency_matrix(transpose=True)[node_ind]._indices()[0,nn.indices].item()

    print("********** COMPUTE IDENTIFIABILITY **********")
    if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed':
        num_identifiable = 0.0
        for node_ind in range(g.number_of_nodes()):
            if node_ind == nn_list[node_ind]:
                num_identifiable+=1
        identifiability_rate = num_identifiable / g.number_of_nodes()
    print('identifiability_rate = {}%'.format(identifiability_rate*100))


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Try to find fixpoint")

    parser.add_argument('dataset', help='choose dataset from: cora, reddit-self-loop, ppi, aids, reddit-binary and imdb-binary')
    parser.add_argument('gcn_model',help='choose GCN model layers from: 2 or 3')
    parser.add_argument('regression_model', help='choose model structure from: slp, mlp or none')
    parser.add_argument('embedding_layer', help='From which layer do you want to check the embedding\'s identifiability?')
    parser.add_argument('regression_metric', help='choose metric for regression and nearest neighbour from: l2 or cos')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")