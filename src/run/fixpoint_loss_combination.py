from src.utils.dataset import load_dataset
from src.utils.train_model_loss_combination import train_citation, train_ppi
from src.models.slp_gcn_loss_combination import SLP_GCN_4node, GCN_2layer

import argparse
import torch
import yaml
import os


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
    if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed, amazon_photos, amazon_computers, coauthors_cs, coauthors_physics':
        g, features, labels, train_mask, valid_mask, test_mask = load_dataset(args)
    elif args.dataset == 'ppi':
        train_dataset, valid_dataset, test_dataset, train_dataloader, valid_dataloader, test_dataloader = load_dataset(args)

    # build network
    print("********** BUILD NETWORK **********")
    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    h_feats = config['hidden_features']

    if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed, amazon_photos, amazon_computers, coauthors_cs, coauthors_physics':
        in_feats = features.shape[1]
        out_feats = torch.max(labels).item() + 1
    elif args.dataset == 'ppi':
        in_feats = train_dataset.features.shape[1]
        out_feats = train_dataset.labels.shape[1]

    if args.loss_weight:
        slp_gcn = SLP_GCN_4node(in_feats, h_feats, out_feats).to(device)
    elif args.without_fixpoint_loss:
        slp_gcn = GCN_2layer(in_feats, h_feats, out_feats).to(device)

    print("********** TRAIN NETWORK **********")
    if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed, amazon_photos, amazon_computers, coauthors_cs, coauthors_physics':
        train_citation(slp_gcn, g, features, labels, train_mask, test_mask, args)
    elif args.dataset == 'ppi':
        train_ppi(slp_gcn, train_dataloader, test_dataloader, args)

    checkpoint_path = '../checkpoints/slp_gcn_' + args.dataset + '.pkl'
    checkpoint_file = os.path.join(os.getcwd(), checkpoint_path)
    torch.save(slp_gcn.state_dict(), checkpoint_file)

if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Try to find fixpoint")

    parser.add_argument('dataset', help='choose dataset from: cora, reddit-self-loop, ppi, aids, reddit-binary and imdb-binary')
    parser.add_argument('--loss_weight', action='store_true', help='choose dataset from: cora, reddit-self-loop, ppi, aids, reddit-binary and imdb-binary')
    parser.add_argument('--without_fixpoint_loss', action='store_true', help='if true add fixpoint loss, else only classification loss')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")

