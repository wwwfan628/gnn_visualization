import torch
import numpy as np
import yaml
import os
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def broyden_method_cora_reddit_ppi(net, graph, features, args):

    writer = SummaryWriter(logdir='../logs/'+args.dataset+'_'+args.method)

    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    max_epoch = config['broyden_method']['max_epoch']  # maximal number of training epochs
    tol = config['broyden_method']['tolerance']

    min_cost_func = float('inf')
    cost_func = float('inf')
    epoch = 0

    H = features.clone().detach().requires_grad_(True).to(device)
    num_elements = H.shape[0] * H.shape[1]
    func_next = net(graph, H) - H
    func_resize_next = func_next.view(num_elements, 1)

    while cost_func > tol and epoch < max_epoch:

        func = func_next  # the matrix we want every element in it equals 0
        cost_func = (torch.sum(torch.abs(func))/float(num_elements)).detach()

        writer.add_scalar('cost function value', cost_func, epoch)

        if cost_func < min_cost_func:
            min_cost_func = cost_func
            H_min_cost_func = H.clone().detach().to(device)
            min_epoch = epoch

        func_resize = func_resize_next

        if epoch == 0:
            gradient_current = torch.zeros([num_elements, num_elements]).to(device)
            for index in np.arange(num_elements):
                func_resize[index].backward(retain_graph=True)
                gradient_current[index, :] = H.grad.view(1, num_elements).clone().detach().to(device)
                H.grad.data.zero_()
            gradient_inverse_current = gradient_current.inverse()
        else:
            if 'bad' in args.method:
                numerator = H_diff - torch.mm(gradient_inverse_former, func_diff).detach()
                multiplier = func_diff.t().detach()
                denominator = (func_diff.norm() ** 2).detach()
            else:
                numerator = H_diff - torch.mm(gradient_inverse_former, func_diff).detach()
                multiplier = torch.mm(H_diff.t(), gradient_inverse_former).detach()
                denominator = torch.mm(torch.mm(H_diff.t(), gradient_inverse_former), func_diff).detach()
            gradient_inverse_current = gradient_inverse_former + torch.mm(numerator,multiplier)/denominator

        H_resize = H.view(num_elements, 1).clone().detach().to(device)
        H_resize_next = H_resize - torch.mm(gradient_inverse_current, func_resize).detach()
        H = H_resize_next.view(H.shape).clone().detach().to(device)

        H_diff = H_resize_next - H_resize
        func_next = net(graph, H) - H
        func_resize_next = func_next.view(num_elements, 1).detach()
        func_diff = func_resize_next - func_resize
        gradient_inverse_former = gradient_inverse_current.clone().detach().to(device)

        if epoch % 50 == 0:    # save current best result
            H_path = '../outputs/H_' + args.dataset + '_' + args.method + '.pkl'
            H_file = os.path.join(os.getcwd(), H_path)
            torch.save(H_min_cost_func, H_file)
            cost_func_path = '../outputs/cost_func_' + args.dataset + '_' + args.method + '.pkl'
            cost_func_file = os.path.join(os.getcwd(), cost_func_path)
            torch.save(min_cost_func, cost_func_file)

        epoch += 1

    if cost_func <= tol:
        print("Fixpoint is found!")
    else:
        print("Reached maximal number of epochs! Current min value of cost function found in epoch {}: {:.4f} ".format(min_epoch, min_cost_func))

    writer.close()

    return H_min_cost_func, min_cost_func


def broyden_method_tu(net, dataset_reduced, args):

    writer = SummaryWriter(logdir='../logs/'+args.dataset+'_'+args.method)

    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    max_epoch = config['broyden_method']['max_epoch']  # maximal number of training epochs
    tol = config['broyden_method']['tolerance']

    H_min_cost_func = []
    fixpoint_found_graph_ind = np.zeros(len(dataset_reduced))
    min_cost_func = np.zeros(len(dataset_reduced))

    for graph_id, data in enumerate(dataset_reduced):

        graph = data[0]
        features = data[0].ndata['feat']

        min_cost_func_graph = float('inf')
        cost_func = float('inf')
        epoch = 0

        H = features.clone().detach().requires_grad_(True).to(device)
        num_elements = H.shape[0] * H.shape[1]
        func_next = net(graph, H) - H
        func_resize_next = func_next.view(num_elements, 1)

        while cost_func > tol and epoch < max_epoch:

            func = func_next  # the matrix we want every element in it equals 0
            cost_func = torch.sum(torch.abs(func))/float(num_elements)

            writer.add_scalar('cost function value of graph' + str(graph_id), cost_func, epoch)

            if cost_func < min_cost_func_graph:
                min_cost_func_graph = cost_func.detach()
                H_min_cost_func_graph = H.clone().detach().to(device)
                min_epoch = epoch

            func_resize = func_resize_next

            if epoch == 0:
                gradient_current = torch.zeros([num_elements, num_elements]).to(device)
                for index in np.arange(num_elements):
                    func_resize[index].backward(retain_graph=True)
                    gradient_current[index, :] = H.grad.view(1, num_elements).clone().detach().to(device)
                    H.grad.data.zero_()
                gradient_inverse_current = gradient_current.inverse()
            else:
                if 'bad' in args.method:
                    numerator = H_diff - torch.mm(gradient_inverse_former, func_diff).detach()
                    multiplier = func_diff.t().detach()
                    denominator = (func_diff.norm() ** 2).detach()
                else:
                    numerator = H_diff - torch.mm(gradient_inverse_former, func_diff).detach()
                    multiplier = torch.mm(H_diff.t(), gradient_inverse_former).detach()
                    denominator = torch.mm(torch.mm(H_diff.t(), gradient_inverse_former), func_diff).detach()
                gradient_inverse_current = gradient_inverse_former + torch.mm(numerator, multiplier) / denominator

            H_resize = H.view(num_elements, 1).clone().detach().to(device)
            H_resize_next = H_resize - torch.mm(gradient_inverse_current, func_resize).detach()
            H = H_resize_next.view(H.shape).clone().detach().to(device)

            H_diff = H_resize_next - H_resize
            func_next = (net(graph, H) - H).detach()
            func_resize_next = func_next.view(num_elements, 1).detach()
            func_diff = func_resize_next - func_resize
            gradient_inverse_former = gradient_inverse_current.clone().detach().to(device)

            epoch += 1

        if cost_func <= tol:
            print("Fixpoint for graph {} is found!".format(graph_id))
            fixpoint_found_graph_ind[graph_id] = True
        else:
            print("Reached maximal number of epochs for graph {}! Current min value of cost function found in epoch {}: {:.4f} ".format(graph_id, min_epoch, min_cost_func_graph))
            fixpoint_found_graph_ind[graph_id] = False

        H_min_cost_func.append(H_min_cost_func_graph)
        min_cost_func[graph_id] = min_cost_func_graph
        # save current result
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


