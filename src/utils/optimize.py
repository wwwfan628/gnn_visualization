import torch
import time

def evaluate_optimization(net, graph, features):
    # used to evaluate the difference of net's input and output
    net.eval()
    with torch.no_grad():
        net_out = net(graph, features)

    F_diff = net_out - features  # the matrix we want every element in it equals 0
    F_diff_abs = torch.abs(F_diff)
    F_cost = torch.sum(torch.sum(F_diff_abs, dim=1))  # cost function: sum of absolute value of each element

    return F_cost


def optmize_fixpoint(net, graph, features):
    H = features  # set input of network as H
    H.requires_grad_(True)  # To compute gradient

    # Optimization criterion
    tol = 0.1  # tolerance: threshold to stop optimization
    epoch = 0  # number of iterations of optimization
    max_epo = 1000000  # max number of iterations
    lr = 1e-3

    optimizer = torch.optim.Adam([H], lr=lr)
    dur = []

    cost_func = 10000  # initialize cost_func to a random value > tolerance

    while cost_func >= tol and epoch < max_epo:
        t0 = time.time()  # start time of current epoch

        F_diff = net(graph, H) - H  # the matrix we want every element in it equals 0
        F_diff_abs = torch.abs(F_diff)
        F_cost = torch.sum(torch.sum(F_diff_abs, dim=1))  # cost function: sum of absolute value of each element

        optimizer.zero_grad()
        F_cost.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            dur.append(time.time() - t0)
            cost_func = evaluate_optimization(net, g, H)
            print("Epoch {:09d} | Cost Function {:.4f} | Time(s) {:.4f}".format(epoch, cost_func, np.mean(dur)))

        epoch += 1

