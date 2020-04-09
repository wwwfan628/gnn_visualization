import torch
import time
import numpy as np
import yaml
import os

def optimize_graph_cora_reddit_ppi(net, graph, features, args):

    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f)

    lr = config['optimize_lr']  # learning rate
    max_epoch = config['optimize_max_epoch']  # maximal number of training epochs
    tol = config['optimize_tolerance']
    min_cost_func = float('inf')

    H = features.clone().detach().requires_grad_(True)  # set input of network as H

    epoch = 0  # current iteration of optimization
    F_cost = float('inf')  # initialize cost_func to a random value > tolerance

    optimizer = torch.optim.Adam([H], lr=lr)
    dur = []

    while F_cost > tol and epoch < max_epoch:
        t0 = time.time()  # start time of current epoch

        F_diff = net(graph, H) - H  # the matrix we want every element in it equals 0
        F_diff_abs = torch.abs(F_diff)
        F_cost = torch.sum(torch.sum(F_diff_abs, dim=1))  # cost function: sum of absolute value of each element

        if F_cost < min_cost_func:
            min_cost_func = F_cost.item()
            H_min_cost_func = H.clone().detach()

        optimizer.zero_grad()
        F_cost.backward()
        optimizer.step()

        if epoch % 100 == 0:
            dur.append(time.time() - t0)
            print("Epoch {:07d} | Cost Function {:.4f} | Time(s) {:.4f}".format(epoch, F_cost, np.mean(dur)))
        epoch += 1

    if F_cost <= tol:
        print("Fixpoint is found!")
    else:
        print("Reached maximal number of epochs! Current min cost function value: {:.4f}".format(min_cost_func))
    return H_min_cost_func


def optimize_graph_tu(net, dataset_reduced, args):

    config_file = os.path.join(os.getcwd(), '../configs/tu.yaml' )
    with open(config_file, 'r') as f:
        config = yaml.load(f)

    lr = config['optimize_lr']  # learning rate
    max_epoch = config['optimize_max_epoch']  # maximal number of training epochs
    tol = config['optimize_tolerance']
    min_cost_func = float('inf')

    H_min_cost_func = []
    fixpoint_found_graph_ind = np.empty(len(dataset_reduced))
    for graph_id, data in enumerate(dataset_reduced):  # loop over each graph
        graph = data[0]
        H = graph.ndata['feat'].clone().detach().requires_grad_(True) # set input of network as H

        epoch = 0  # current iteration of optimization
        F_cost = float('inf')  # initialize cost_func to a random value > tolerance

        optimizer = torch.optim.Adam([H], lr=lr)
        dur = []

        while F_cost > tol and epoch < max_epoch:
            t0 = time.time()  # start time of current epoch

            F_diff = net(graph, H) - H  # the matrix we want every element in it equals 0
            F_diff_abs = torch.abs(F_diff)
            F_cost = torch.sum(torch.sum(F_diff_abs, dim=1))  # cost function: sum of absolute value of each element

            if F_cost < min_cost_func:
                min_cost_func = F_cost.item()
                H_min_cost_func_current_graph = H.clone().detach()

            optimizer.zero_grad()
            F_cost.backward()
            optimizer.step()

            if epoch % 100 == 0:
                dur.append(time.time() - t0)
                print("Graph ID {:06d} | Epoch {:07d} | Cost Function {:.4f} | Time(s) {:.4f}".format(graph_id, epoch, F_cost, np.mean(dur)))
            epoch += 1

        if F_cost <= tol:
            print("Fixpoint for graph {} is found!".format(graph_id))
            fixpoint_found_graph_ind[graph_id] = True
        else:
            print("Reached maximal number of epochs! Current min cost function value for graph {}: {:.4f}".format(min_cost_func))
            fixpoint_found_graph_ind[graph_id] = False

        H_min_cost_func.append(H_min_cost_func_current_graph)

    return H_min_cost_func, fixpoint_found_graph_ind