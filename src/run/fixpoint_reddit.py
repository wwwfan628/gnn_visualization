from src.utils.dataset import load_reddit_data
from src.utils.train import train_net
from src.utils.train import load_parameters
from src.utils.optimize import optmize_fixpoint
from src.models.mlp_gcn import MLP_GCN
from src.models.mlp import MLP
from src.models.gcn import GCN

import argparse
import torch


def get_parser():
    # get parameters for program
    parser = argparse.ArgumentParser()
    parser.add_argument('--tol', type=float, default=0.1, help='tolerance of optimization')
    parser.add_argument('--epoch_optimize', type=int, default=1000000, help='number of optimization epochs')
    parser.add_argument('--epoch_train', type=int, default=500, help='number of training epochs')
    parser.add_argument('--lr_optimize', type=float, default=1e-3, help='learning rate of optimization')
    parser.add_argument('--lr_train', type=float, default=1e-3, help='learning rate of training')
    return parser


def main(args):
    # load cora dataset

    g, features, labels, train_mask, test_mask = load_reddit_data()

    # train network and save the parameters of the trained network
    mlp_gcn = MLP_GCN(602, 320, 160, 72, 72, 72, 72, 41)
    train_net(mlp_gcn, g, features, labels, train_mask, test_mask, args)
    file = 'parameters_reddit.pkl'
    torch.save(mlp_gcn.state_dict(), file)

    # reduce dimension of nodes'features
    mlp = MLP(602, 320, 160, 72)
    model_dict = load_parameters(file, mlp)
    mlp.load_state_dict(model_dict)
    mlp.eval()
    with torch.no_grad():
        features_reduced = mlp(features)

    # GCN
    gcn = GCN(72, 72, 72, 72)
    model_dict = load_parameters(file, gcn)
    gcn.load_state_dict(model_dict)

    H = optmize_fixpoint(gcn, g, features_reduced, args)
    H_file = 'H_reddit.pkl'
    torch.save(H, H_file)


if __name__ == '__main__':

    # get parameters
    parser = get_parser()
    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")
