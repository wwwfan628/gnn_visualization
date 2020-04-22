import torch
import time
import numpy as np
import yaml
import os
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def newton_method_cora_reddit_ppi(net, graph, features, args):

    writer = SummaryWriter('./logs/' + args.dataset + '_' + args.method)

    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    max_epoch = config['newton_method']['max_epoch']  # maximal number of training epochs
    tol = config['newton_method']['tolerance']
    min_cost_func = float('inf')
    cost_func = float('inf')
    epoch = 0

    H = features.clone().detach().requires_grad_(True).to(device)
    num_elements = H.shape[0] * H.shape[1]
    #H_diff = float('inf')  # Difference between H before and after one iteration

    # dur = []
    while cost_func > tol and epoch < max_epoch:

        # t0 = time.time()  # strat time of current epoch

        func = net(graph, H) - H  # the matrix we want every element in it equals 0
        cost_func = torch.sum(torch.abs(func))/float(num_elements)

        writer.add_scalar('cost function value', cost_func, epoch)

        if cost_func < min_cost_func:
            min_cost_func = cost_func
            H_min_cost_func = H.clone().detach().to(device)
            min_epoch = epoch

        func_resize = func.view(num_elements, 1)

        gradient = torch.zeros([num_elements, num_elements]).to(device)

        for index in np.arange(num_elements):
            func_resize[index].backward(retain_graph=True)
            gradient[index, :] = H.grad.view(1, num_elements).clone().detach().to(device)
            H.grad.data.zero_()

        H_resize = H.view(num_elements, 1).clone().detach().to(device)
        H_resize_new = H_resize - torch.mm(gradient.inverse(), func_resize)
        H = H_resize_new.view(H.shape).clone().detach().requires_grad_(True).to(device)

        # H_diff = torch.sum(torch.abs(H_resize - H_resize_new))
        # if epoch % 1 == 0:
        #     dur.append(time.time() - t0)
        #     print("Epoch {} | Current Function {:.4f} | Diff {:.4f} | Time(s) {:.4f}".format(epoch, cost_func, H_diff, np.mean(dur)))

        epoch += 1

    if cost_func <= tol:
        print("Fixpoint is found!")
    else:
        print("Reached maximal number of epochs! Current min value of cost function found in epoch {}: {:.4f} ".format(min_epoch, min_cost_func))

    writer.close()

    return H_min_cost_func, min_cost_func


def newton_method_tu(net, dataset_reduced, args):

    writer = SummaryWriter('./logs/' + args.dataset + '_' + args.method)

    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    max_epoch = config['newton_method']['max_epoch']  # maximal number of training epochs
    tol = config['newton_method']['tolerance']

    H_min_cost_func = []
    fixpoint_found_graph_ind = np.empty(len(dataset_reduced))
    min_cost_func = np.empty(len(dataset_reduced))

    for graph_id, data in enumerate(dataset_reduced):

        min_cost_func_graph = float('inf')

        graph = data[0]
        features = data[0].ndata['feat']
        H = features.clone().detach().requires_grad_(True).to(device)
        num_elements = H.shape[0] * H.shape[1]
        # H_diff = float('inf')  # Difference between H before and after one iteration
        cost_func = float('inf')
        epoch = 0

        # dur = []
        while cost_func > tol and epoch < max_epoch:

            # t0 = time.time()  # strat time of current epoch

            func = net(graph, H) - H  # the matrix we want every element in it equals 0
            cost_func = torch.sum(torch.abs(func))/float(num_elements)

            writer.add_scalar('cost function value of graph' + graph_id, cost_func, epoch)

            if cost_func < min_cost_func_graph:
                min_cost_func_graph = cost_func
                H_min_cost_func_graph = H.clone().detach().to(device)
                min_epoch = epoch

            func_resize = func.view(num_elements, 1)

            gradient = torch.zeros([num_elements, num_elements]).to(device)

            for index in np.arange(num_elements):
                func_resize[index].backward(retain_graph=True)
                gradient[index, :] = H.grad.view(1, num_elements).clone().detach().to(device)
                H.grad.data.zero_()

            H_resize = H.view(num_elements, 1).clone().detach().to(device)
            H_resize_new = H_resize - torch.mm(gradient.inverse(), func_resize)
            H = H_resize_new.view(H.shape).clone().detach().requires_grad_(True).to(device)

            # H_diff = torch.sum(torch.abs(H_resize - H_resize_new))
            # if epoch % 1 == 0:
            #     dur.append(time.time() - t0)
            #     print("Graph ID {} | Epoch {} | Current Function {:.4f} | Diff {:.4f} | Time(s) {:.4f}".format(graph_id, epoch, func_abs_sum, H_diff, np.mean(dur)))
            epoch += 1

        if cost_func <= tol:
            print("Fixpoint for graph {} is found!".format(graph_id))
            fixpoint_found_graph_ind[graph_id] = True
        else:
            print("Reached maximal number of epochs for graph {}! Current min value of cost function found in epoch {}: {:.4f} ".format(graph_id, min_epoch, min_cost_func_graph))
            fixpoint_found_graph_ind[graph_id] = False

        H_min_cost_func.append(H_min_cost_func_graph)
        min_cost_func[graph_id] = min_cost_func_graph

    num_found = fixpoint_found_graph_ind[fixpoint_found_graph_ind == True].shape[0]
    print("The number of graphs successfully finding fixpoint: {} ".format(num_found))

    writer.close()

    return H_min_cost_func, fixpoint_found_graph_ind, min_cost_func