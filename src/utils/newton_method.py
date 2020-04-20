import torch
import time
import numpy as np
import yaml
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def newton_method_cora_reddit_ppi(net, graph, features, args):

    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    max_epoch = config['newton_method']['max_epoch']  # maximal number of training epochs
    tol = config['newton_method']['tolerance']
    min_func_abs_sum = float('inf')

    H = features.clone().detach().requires_grad_(True)
    H_diff = float('inf')  # Difference between H before and after one iteration
    epoch = 0

    dur = []
    while H_diff > tol and epoch < max_epoch:

        t0 = time.time()  # strat time of current epoch

        func = net(graph, H) - H  # the matrix we want every element in it equals 0
        func_abs_sum = torch.sum(torch.sum(torch.abs(func), dim=1))
        if func_abs_sum < min_func_abs_sum:
            min_func_abs_sum = func_abs_sum
            H_min_func_abs_sum = H.clone().detach()
            min_epoch = epoch

        num_elements = features.shape[0] * features.shape[1]
        func_resize = func.view(num_elements, 1)

        gradient = torch.zeros([num_elements, num_elements])

        for index in np.arange(num_elements):
            func_resize[index].backward(retain_graph=True)
            gradient[index, :] = H.grad.view(1, num_elements).clone().detach()
            H.grad.data.zero_()

        H_resize = H.view(num_elements, 1).clone().detach()
        H_resize_new = H_resize - torch.mm(gradient.inverse(), func_resize)
        H = H_resize_new.view(H.shape).clone().detach().requires_grad_(True)

        H_diff = torch.sum(torch.abs(H_resize - H_resize_new))
        if epoch % 1 == 0:
            dur.append(time.time() - t0)
            print("Epoch {} | Current Function {:.4f} | Diff {:.4f} | Time(s) {:.4f}".format(epoch, func_abs_sum, H_diff, np.mean(dur)))
        epoch += 1

    if H_diff <= tol:
        print("Fixpoint is found!")
    else:
        print("Reached maximal number of epochs! Current min value of cost function found in epoch {}: {:.4f} ".format(min_epoch, min_func_abs_sum))

    return H_min_func_abs_sum


def newton_method_tu(net, dataset_reduced, args):

    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    max_epoch = config['newton_method']['max_epoch']  # maximal number of training epochs
    tol = config['newton_method']['tolerance']

    H_min_func_abs_sum = []
    fixpoint_found_graph_ind = np.empty(len(dataset_reduced))
    for graph_id, data in enumerate(dataset_reduced):

        min_func_abs_sum_graph = float('inf')

        graph = data[0]
        features = data[0].ndata['feat']
        H = features.clone().detach().requires_grad_(True)
        H_diff = float('inf')  # Difference between H before and after one iteration
        epoch = 0

        dur = []
        while H_diff > tol and epoch < max_epoch:

            t0 = time.time()  # strat time of current epoch

            func = net(graph, H) - H  # the matrix we want every element in it equals 0
            func_abs_sum = torch.sum(torch.sum(torch.abs(func), dim=1))
            if func_abs_sum < min_func_abs_sum_graph:
                min_func_abs_sum_graph = func_abs_sum
                H_min_func_abs_sum_graph = H.clone().detach()
                min_epoch = epoch

            num_elements = features.shape[0] * features.shape[1]
            func_resize = func.view(num_elements, 1)

            gradient = torch.zeros([num_elements, num_elements])

            for index in np.arange(num_elements):
                func_resize[index].backward(retain_graph=True)
                gradient[index, :] = H.grad.view(1, num_elements).clone().detach()
                H.grad.data.zero_()

            H_resize = H.view(num_elements, 1).clone().detach()
            H_resize_new = H_resize - torch.mm(gradient.inverse(), func_resize)
            H = H_resize_new.view(H.shape).clone().detach().requires_grad_(True)

            H_diff = torch.sum(torch.abs(H_resize - H_resize_new))
            if epoch % 1 == 0:
                dur.append(time.time() - t0)
                print("Epoch {} | Current Function {:.4f} | Diff {:.4f} | Time(s) {:.4f}".format(epoch, func_abs_sum, H_diff, np.mean(dur)))
            epoch += 1

        if H_diff <= tol:
            print("Fixpoint for graph {} is found!".format(graph_id))
            fixpoint_found_graph_ind[graph_id] = True
        else:
            print("Reached maximal number of epochs for graph {}! Current min value of cost function found in epoch {}: {:.4f} ".format(graph_id, min_epoch, min_func_abs_sum_graph))
            fixpoint_found_graph_ind[graph_id] = False

        H_min_func_abs_sum.append(H_min_func_abs_sum_graph)

    num_found = fixpoint_found_graph_ind[fixpoint_found_graph_ind == True].shape[0]
    print("The number of graphs successfully finding fixpoint: {} ".format(num_found))
    return H_min_func_abs_sum, fixpoint_found_graph_ind