import torch

from src.utils.load_data import load_cora_data
from src.utils.train import train_net
from src.models.mlp_gcn import MLP_GCN
from src.models.mlp import MLP
from src.models.gcn import GCN

import argparse
import torch


def get_parser():
    # get parameters for program
    parser = argparse.ArgumentParser()
    parser.add_argument('--tol', type=float, default=0.1, nargs='?')
    parser.add_argument('--epoch_optimize', type=int, default=1000000, nargs='?')
    parser.add_argument('--epoch_train', type=int, default=500, nargs='?')
    parser.add_argument('--lr_optimize', type=float, default=1e-3, nargs='?')
    parser.add_argument('--lr_train', type=float, default=1e-3, nargs='?')
    return parser


if __name__ == '__main__':

    # get parameters
    parser = get_parser()
    args = parser.parse_args()

    # load cora dataset
    g, features, labels, train_mask, test_mask = load_cora_data()

    #

