from src.utils.dataset import load_dataset
from src.utils.train_model import train_cora_reddit,train_ppi
from src.models.gcn import GCN_Baseline, GCN_Baseline_3Layers

import argparse
import torch
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
    if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed':
        g, features, labels, train_mask, valid_mask, test_mask = load_dataset(args)
    elif args.dataset == 'ppi':
        train_dataset, valid_dataset, train_dataloader, valid_dataloader = load_dataset(args)
    elif 'tu' in args.dataset:
        statistics, train_dataset, valid_dataset, train_dataloader, valid_dataloader = load_dataset(args)

    # build network
    print("********** BUILD NETWORK **********")
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

    gcn_baseline = GCN_Baseline_3Layers(in_feats, h_feats, out_feats).to(device)

    print("********** TRAIN NETWORK **********")
    if args.dataset in 'cora, reddit-self-loop, citeseer, pubmed':
        train_cora_reddit(gcn_baseline, g, features, labels, train_mask, test_mask, args)
    elif args.dataset == 'ppi':
        train_ppi(gcn_baseline, train_dataloader, valid_dataloader, args)

    checkpoint_path = '../checkpoints/slp_gcn_' + args.dataset + '.pkl'
    checkpoint_file = os.path.join(os.getcwd(), checkpoint_path)
    torch.save(gcn_baseline.state_dict(), checkpoint_file)

if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Try to find fixpoint")

    parser.add_argument('dataset', help='choose dataset from: cora, reddit-self-loop, ppi, aids, reddit-binary and imdb-binary')
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")

