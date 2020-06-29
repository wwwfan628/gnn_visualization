import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import dgl
import argparse
from dgl.data import load_data
from dgl import DGLGraph
import networkx as nx
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('dataset.py running on {}!'.format(device))

class SubgraphSteadyStateOperator(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(SubgraphSteadyStateOperator, self).__init__()
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)

    def forward(self, subg):
        def message_func(edges):
            x = edges.src['x']
            h = edges.src['h']
            return {'m': torch.cat([x, h], dim=1)}

        def reduce_func(nodes):
            m = torch.sum(nodes.mailbox['m'], dim=1)
            z = torch.cat([nodes.data['x'], m], dim=1)
            return {'h': self.linear2(F.relu(self.linear1(z)))}

        subg.block_compute(0, message_func, reduce_func)
        return subg.layers[-1].data['h']

class Predictor(nn.Module):
    def __init__(self, n_hidden, n_output):
        super(Predictor, self).__init__()
        self.linear1 = nn.Linear(n_hidden, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_output)  ## classifier

    def forward(self, h):
        return self.linear2(F.relu(self.linear1(h)))

def update_parameters_subgraph(subg, steady_state_operator, predictor, optimizer):
    n = subg.layer_size(-1)
    steady_state_operator.train()
    predictor.train()
    steady_state_operator(subg)
    z = predictor(subg.layers[-1].data['h'])
    y_predict = F.log_softmax(z, 1)
    y = subg.layers[-1].data['y']  # label
    loss = F.nll_loss(y_predict, y) * 1.0 / n

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  # TODO: divide gradients by the number of labelled nodes?

def update_embeddings_subgraph(g, steady_state_operator, alpha=0.1):
    # Note that we are only updating the embeddings of seed nodes here.
    # The reason is that only the seed nodes have ample information
    # from neighbors, especially if the subgraph is small (e.g. 1-hops)
    prev_h = g.layers[-1].data['h']
    next_h = steady_state_operator(g)
    g.layers[-1].data['h'] = (1 - alpha) * prev_h + alpha * next_h

def train_on_subgraphs(g, label_nodes, batch_size, neigh_expand, steady_state_operator, predictor, optimizer, n_embedding_updates = 8, n_parameter_updates = 5, alpha = 0.1):
    # To train SSE, we create two subgraph samplers with the
    # `NeighborSampler` API for each phase.

    # The first phase samples from all vertices in the graph.
    sampler = dgl.contrib.sampling.NeighborSampler(g, batch_size, neigh_expand, num_hops=1, shuffle=True)
    sampler_iter = iter(sampler)

    # The second phase only samples from labeled vertices.
    sampler_train = dgl.contrib.sampling.NeighborSampler(g, batch_size, neigh_expand, seed_nodes=label_nodes, num_hops=1, shuffle=True)
    sampler_train_iter = iter(sampler_train)

    for i in range(n_embedding_updates):
        subg = next(sampler_iter)
        # Currently, subgraphing does not copy or share features
        # automatically.  Therefore, we need to copy the node
        # embeddings of the subgraph from the parent graph with
        # `copy_from_parent()` before computing...
        subg.copy_from_parent()
        # print('!!!')
        # print('Before update')
        # print(subg.layers[-1].data['h'])
        update_embeddings_subgraph(subg, steady_state_operator, alpha=alpha)
        # print('After update')
        # print(subg.layers[-1].data['h'])
        # ... and copy them back to the parent graph.
        g.ndata['h'][subg.layer_parent_nid(-1)] = subg.layers[-1].data['h'].detach()
    for i in range(n_parameter_updates):
        try:
            subg = next(sampler_train_iter)
        except:
            break
        # Again we need to copy features from parent graph
        subg.copy_from_parent()
        update_parameters_subgraph(subg, steady_state_operator, predictor, optimizer)
        # We don't need to copy the features back to parent graph.

def test(g, test_nodes, predictor):
    predictor.eval()
    with torch.no_grad():
        z = predictor(g.ndata['h'][test_nodes])
        _, indices = torch.max(z, dim=1)
        y = g.ndata['y'][test_nodes]
        accuracy = torch.sum(indices == y) * 1.0 / len(test_nodes)
        y_predict = F.log_softmax(z, 1)
        loss = F.nll_loss(y_predict, y) * 1.0
        return accuracy.item(), loss

def load_citation(args):
    data = load_data(args)
    features = torch.FloatTensor(data.features).to(device)
    labels = torch.LongTensor(data.labels).to(device)
    train_mask = torch.BoolTensor(data.train_mask).to(device)
    test_mask = torch.BoolTensor(data.test_mask).to(device)
    valid_mask = torch.BoolTensor(data.val_mask).to(device)
    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, train_mask, valid_mask, test_mask

def main(args):
    # load dataset
    g, features, labels, train_mask, valid_mask, test_mask = load_citation(args)
    g.readonly(True)
    g.ndata['x'] = features.clone().detach()
    g.ndata['h'] = torch.zeros((g.number_of_nodes(), args.n_hidden))
    g.ndata['y'] = labels.clone().detach()

    nodes_train = np.arange(features.shape[0])[train_mask]
    nodes_test = np.arange(features.shape[0])[test_mask]

    n_input = features.shape[1] * 2 + args.n_hidden
    n_output = torch.max(labels).item() + 1
    subgraph_steady_state_operator = SubgraphSteadyStateOperator(n_input, args.n_hidden)
    predictor = Predictor(args.n_hidden, n_output)
    params = list(subgraph_steady_state_operator.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    best_accuracy = -1
    best_loss = float('inf')
    patience = 40

    for i in range(args.n_epochs):
        train_on_subgraphs(g, nodes_train, args.batch_size, args.neigh_expand, subgraph_steady_state_operator,
            predictor, optimizer, args.n_embedding_updates, args.n_parameter_updates, args.alpha)
        accuracy_train, loss_train = test(g, nodes_train, predictor)
        accuracy_test, loss_test = test(g, nodes_test, predictor)
        print("Iter {:05d} | Train acc {:.4f} | Test acc {:.4f}".format(i, accuracy_train, accuracy_test))
        # early stop
        if accuracy_test > best_accuracy or best_loss > loss_test:
            best_accuracy = np.max((accuracy_test, best_accuracy))
            best_loss = np.min((best_loss, loss_test))
            cur_step = 0
        else:
            cur_step += 1
            if cur_step == patience:
                break


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Try to find fixpoint")

    parser.add_argument('--dataset', default='cora', help='choose dataset from: cora, pubmed, citeseer')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_hidden', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--n_embedding_updates', type=int, default=8)
    parser.add_argument('--n_parameter_updates', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument("--neigh_expand", type=int, default=8, help="the number of neighbors to sample.")

    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")