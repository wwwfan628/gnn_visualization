import torch

from src.utils.load_data import load_cora_data
from src.utils.train import train_net

import argparse
import torch


def get_parser():
    # set parameters for optimization
    parser = argparse.ArgumentParser()
    parser.add_argument('--tol', type=float, default=0.1, nargs='?')
    parser.add_argument('--max_epoch', type=int, default=1000000, nargs='?')
    parser.add_argument('--lr', type=float, default=1e-3, nargs='?')
    return parser


if __name__ == '__main__':

    # set parameters for optimization
    parser = get_parser()
    args = parser.parse_args()

    # load cora dataset
    g, features, labels, train_mask, test_mask = load_cora_data()

    #

