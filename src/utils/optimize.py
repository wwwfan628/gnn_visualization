import torch
import time
import numpy as np
import yaml
import os
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def optimize_graph_cora_reddit_ppi(net, graph, features, args):

    writer = SummaryWriter(logdir='./logs/'+args.dataset+'_'+args.method)

    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    lr = config['graph_optimization']['optimize_lr']  # learning rate
    max_epoch = config['graph_optimization']['optimize_max_epoch']  # maximal number of training epochs
    tol = config['graph_optimization']['optimize_tolerance']
    min_cost_func = float('inf')

    H = features.clone().detach().requires_grad_(True).to(device)  # set input of network as H
    num_elements = float(H.shape[0] * H.shape[1])

    epoch = 0  # current iteration of optimization
    F_cost = float('inf')  # initialize cost_func to a random value > tolerance

    optimizer = torch.optim.Adam([H], lr=lr)
    #dur = []

    while F_cost > tol and epoch < max_epoch:
        #t0 = time.time()  # start time of current epoch

        F_diff = net(graph, H) - H  # the matrix we want every element in it equals 0
        F_diff_abs = torch.abs(F_diff)
        F_cost = torch.sum(F_diff_abs)/num_elements  # cost function: sum of absolute value of each element

        writer.add_scalar('cost function value', F_cost, epoch)

        if F_cost < min_cost_func:
            min_cost_func = F_cost.item()
            H_min_cost_func = H.clone().detach()

        optimizer.zero_grad()
        F_cost.backward()
        optimizer.step()

        if epoch % 100 == 0:    # save current best results
        #     dur.append(time.time() - t0)
        #     print("Epoch {:07d} | Cost Function {:.4f} | Time(s) {:.4f}".format(epoch, F_cost, np.mean(dur)))
            H_path = '../outputs/H_' + args.dataset + '_' + args.method + '.pkl'
            H_file = os.path.join(os.getcwd(), H_path)
            torch.save(H_min_cost_func, H_file)
            cost_func_path = '../outputs/cost_func_' + args.dataset + '_' + args.method + '.pkl'
            cost_func_file = os.path.join(os.getcwd(), cost_func_path)
            torch.save(min_cost_func, cost_func_file)

        epoch += 1

    if F_cost <= tol:
        print("Fixpoint is found!")
    else:
        print("Reached maximal number of epochs! Current min cost function value: {:.4f}".format(min_cost_func))

    writer.close()

    return H_min_cost_func, min_cost_func


def optimize_graph_tu(net, dataset_reduced, args):

    writer = SummaryWriter(logdir='./logs/'+args.dataset+'_'+args.method)

    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    lr = config['graph_optimization']['optimize_lr']  # learning rate
    max_epoch = config['graph_optimization']['optimize_max_epoch']  # maximal number of training epochs
    tol = config['graph_optimization']['optimize_tolerance']

    H_min_cost_func = []
    fixpoint_found_graph_ind = np.zeros(len(dataset_reduced))
    min_cost_func = np.zeros(len(dataset_reduced))

    for graph_id, data in enumerate(dataset_reduced):  # loop over each graph

        min_cost_func_graph = float('inf')
        graph = data[0]
        H = graph.ndata['feat'].clone().detach().requires_grad_(True).to(device) # set input of network as H
        num_elements = float(H.shape[0] * H.shape[1])

        epoch = 0  # current iteration of optimization
        F_cost = float('inf')  # initialize cost_func to a random value > tolerance

        optimizer = torch.optim.Adam([H], lr=lr)
        #dur = []

        while F_cost > tol and epoch < max_epoch:
            #t0 = time.time()  # start time of current epoch

            F_diff = net(graph, H) - H  # the matrix we want every element in it equals 0
            F_diff_abs = torch.abs(F_diff)
            F_cost = torch.sum(F_diff_abs)/num_elements  # cost function: sum of absolute value of each element

            writer.add_scalar('cost function value of graph' + str(graph_id), F_cost, epoch)

            if F_cost < min_cost_func_graph:
                min_cost_func_graph = F_cost.item()
                H_min_cost_func_graph = H.clone().detach().to(device)

            optimizer.zero_grad()
            F_cost.backward()
            optimizer.step()

            # if epoch % 100 == 0:
            #     dur.append(time.time() - t0)
            #     print("Graph ID {:06d} | Epoch {:07d} | Cost Function {:.4f} | Time(s) {:.4f}".format(graph_id, epoch, F_cost, np.mean(dur)))

            epoch += 1

        if F_cost <= tol:
            print("Fixpoint for graph {} is found!".format(graph_id))
            fixpoint_found_graph_ind[graph_id] = True
        else:
            print("Reached maximal number of epochs! Current min cost function value for graph {}: {:.4f}".format(graph_id, min_cost_func_graph))
            fixpoint_found_graph_ind[graph_id] = False

        H_min_cost_func.append(H_min_cost_func_graph)
        min_cost_func[graph_id] = min_cost_func_graph

        # save current results
        H_path = '../outputs/H_' + args.dataset + '_' + args.method + '.pkl'
        H_file = os.path.join(os.getcwd(), H_path)
        torch.save(H_min_cost_func, H_file)
        cost_func_path = '../outputs/cost_func_' + args.dataset + '_' + args.method + '.pkl'
        cost_func_file = os.path.join(os.getcwd(), cost_func_path)
        torch.save(min_cost_func, cost_func_file)
        indices_path = '../outputs/indices_' + args.dataset + '_' + args.method + '.pkl'
        indices_file = os.path.join(os.getcwd(), indices_path)
        torch.save(fixpoint_found_graph_ind, indices_file)

    num_found = fixpoint_found_graph_ind[fixpoint_found_graph_ind == True].shape[0]
    print("The number of graphs successfully finding fixpoint: {} ".format(num_found))

    writer.close()

    return H_min_cost_func, fixpoint_found_graph_ind, min_cost_func


def optimize_node_cora_reddit_ppi(net, graph, features, args):

    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    lr = config['node_optimization']['optimize_lr']  # learning rate
    max_epoch = config['node_optimization']['optimize_max_epoch']  # maximal number of training epochs
    tol = config['node_optimization']['optimize_tolerance']

    H_min_cost_func = torch.zeros(features.shape).to(device)
    fixpoint_found_node_ind = torch.empty(features.shape[0]).to(device)
    for node_id, node_features in enumerate(features):  # optimization for each node
        min_cost_func = float('inf')  # reset min cost function value as 0 for new node
        epoch = 0  # current iteration of optimization
        F_cost = float('inf')  # initialize cost_func to a random value > tolerance

        node_features.requires_grad_(True)
        H = torch.empty(features.shape).requires_grad_(False).to(device)
        H[:node_id+1] = features[:node_id+1].clone().detach().requires_grad_(False).to(device)
        H[node_id+1:] = features[node_id+1:].clone().detach().requires_grad_(False).to(device)

        optimizer = torch.optim.Adam([node_features], lr=lr)
        dur = []

        while F_cost > tol and epoch < max_epoch:
            t0 = time.time()  # start time of current epoch
            H[node_id] = node_features.to(device)
            F_diff = net(graph, H) - H  # the matrix we want every element in it equals 0
            F_diff_abs = torch.abs(F_diff)
            F_cost = torch.sum(F_diff_abs, dim=1)[node_id]  # cost function: sum of absolute value of each element

            if F_cost < min_cost_func:
                min_cost_func = F_cost.item()
                H_min_cost_func_node = node_features.clone().detach()

            optimizer.zero_grad()
            F_cost.backward(retain_graph=True)
            optimizer.step()

            if epoch % 100 == 0:
                dur.append(time.time() - t0)
                print("Epoch {:07d} | Cost Function {:.4f} | Time(s) {:.4f}".format(epoch, F_cost, np.mean(dur)))
            epoch += 1

        if F_cost <= tol:
            print("Fixpoint is found!")
            fixpoint_found_node_ind[node_id] = True
        else:
            print("Reached maximal number of epochs! Current min cost function value: {:.4f}".format(min_cost_func))
            fixpoint_found_node_ind[node_id] = False

        H_min_cost_func[node_id, :] = H_min_cost_func_node

    num_found = fixpoint_found_node_ind[fixpoint_found_node_ind==True].shape[0]
    print("The number of nodes successfully finding fixpoint: {} ".format(num_found))
    return H_min_cost_func, fixpoint_found_node_ind


def optimize_node_tu(net, dataset_reduced, args):

    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    lr = config['node_optimization']['optimize_lr']  # learning rate
    max_epoch = config['node_optimization']['optimize_max_epoch']  # maximal number of training epochs
    tol = config['node_optimization']['optimize_tolerance']

    H_min_cost_func = []
    num_found = 0

    for graph_id, data in enumerate(dataset_reduced):  # loop over each graph

        graph = data[0]
        features = data[0].ndata['feat']

        H_min_cost_func_graph = torch.zeros(features.shape).to(device)
        fixpoint_found_node_ind = torch.empty(features.shape[0]).to(device)

        for node_id, node_features in enumerate(features):  # optimization for each node
            min_cost_func = float('inf')  # reset min cost function value as 0 for new node
            epoch = 0  # current iteration of optimization
            F_cost = float('inf')  # initialize cost_func to a random value > tolerance

            node_features.requires_grad_(True)
            H = torch.empty(features.shape).requires_grad_(False).to(device)
            H[:node_id+1] = features[:node_id+1].clone().detach().requires_grad_(False).to(device)
            H[node_id+1:] = features[node_id+1:].clone().detach().requires_grad_(False).to(device)


            optimizer = torch.optim.Adam([node_features], lr=lr)
            dur = []

            while F_cost > tol and epoch < max_epoch:
                t0 = time.time()  # start time of current epoch
                H[node_id] = node_features.to(device)
                F_diff = net(graph, H) - H  # the matrix we want every element in it equals 0
                F_diff_abs = torch.abs(F_diff)
                F_cost = torch.sum(F_diff_abs, dim=1)[node_id]  # cost function: sum of absolute value of each element

                if F_cost < min_cost_func:
                    min_cost_func = F_cost.item()
                    H_min_cost_func_node = node_features.clone().detach()

                optimizer.zero_grad()
                F_cost.backward(retain_graph=True)
                optimizer.step()

                if epoch % 100 == 0:
                    dur.append(time.time() - t0)
                    print("Graph ID {:06d} | Node ID {} | Epoch {:07d} | Cost Function {:.4f} | Time(s) {:.4f}".format(graph_id, node_id, epoch, F_cost, np.mean(dur)))
                epoch += 1

            if F_cost <= tol:
                print("Fixpoint for graph {} node {} is found!".format(graph_id, node_id))
                fixpoint_found_node_ind[node_id] = True
            else:
                print("Reached maximal number of epochs! Current min cost function value for graph {} node {}: {:.4f}".format(graph_id, node_id, min_cost_func))
                fixpoint_found_node_ind[node_id] = False

            H_min_cost_func_graph[node_id, :] = H_min_cost_func_node

        H_min_cost_func.append(H_min_cost_func_graph)
        num_found = num_found + fixpoint_found_node_ind[fixpoint_found_node_ind==True].shape[0]

    print("The total number of nodes successfully finding fixpoint: {} ".format(num_found))
    return H_min_cost_func, num_found