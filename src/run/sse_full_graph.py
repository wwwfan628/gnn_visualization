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

class FullgraphSteadyStateOperator(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(FullgraphSteadyStateOperator, self).__init__()
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)

    def forward(self, g):
        def message_func(edges):
            x = edges.src['x']
            h = edges.src['h']
            return {'m': torch.cat([x, h], dim=1)}

        def reduce_func(nodes):
            m = torch.sum(nodes.mailbox['m'], dim=1)
            z = torch.cat([nodes.data['x'], m], dim=1)
            return {'h': self.linear2(F.relu(self.linear1(z)))}

        g.update_all(message_func, reduce_func)
        return g.ndata['h']

class Predictor(nn.Module):
    def __init__(self, n_hidden, n_output):
        super(Predictor, self).__init__()
        self.linear1 = nn.Linear(n_hidden, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_output)  ## classifier

    def forward(self, h):
        return self.linear2(F.relu(self.linear1(h)))

def update_parameters_fullgraph(g, label_nodes, steady_state_operator, predictor, optimizer):
    steady_state_operator.train()
    predictor.train()
    steady_state_operator(g)
    z = predictor(g.ndata['h'][label_nodes])
    y_predict = F.log_softmax(z, 1)
    y = g.ndata['y'][label_nodes] # label
    loss = F.nll_loss(y_predict, y)

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()  # TODO: divide gradients by the number of labelled nodes?
    return loss.item()

def update_embeddings(g, steady_state_operator, alpha=0.1):
    prev_h = g.ndata['h']
    next_h = steady_state_operator(g)
    g.ndata['h'] = (1 - alpha) * prev_h + alpha * next_h

def train_on_fullgraph(g, label_nodes, steady_state_operator, predictor, optimizer, n_embedding_updates = 8, n_parameter_updates = 5, alpha = 0.1):
    # The first phase
    for i in range(n_embedding_updates):
        update_embeddings(g, steady_state_operator, alpha=alpha)
    # The second phase
    for i in range(n_parameter_updates):
        loss = update_parameters_fullgraph(g, label_nodes, steady_state_operator, predictor, optimizer)
    return loss

def test(g, test_nodes, predictor):
    predictor.eval()
    with torch.no_grad():
        z = predictor(g.ndata['h'][test_nodes])
        _, indices = torch.max(z, dim=1)
        y = g.ndata['y'][test_nodes]
        accuracy = torch.sum(indices == y)*1.0 / len(test_nodes)
        return accuracy.item(), z

def load_citation(args):
    data = load_data(args)
    features = torch.FloatTensor(data.features).to(device)
    labels = torch.LongTensor(data.labels).to(device)
    train_mask = torch.BoolTensor(data.train_mask).to(device)
    test_mask = torch.BoolTensor(data.test_mask).to(device)
    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, train_mask, test_mask

def main(args):
    fullgraph_steady_state_operator = FullgraphSteadyStateOperator(args.n_input,args.n_hidden)
    predictor = Predictor(args.n_hidden, args.n_output)
    params = list(fullgraph_steady_state_operator.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # load dataset
    g, features, labels, train_mask, test_mask = load_citation(args)
    g.ndata['x'] = features.clone().detach()
    g.ndata['h'] = torch.normal(0, 1, size=(g.number_of_nodes(), args.n_hidden)).requires_grad_(False)
    g.ndata['y'] = labels.clone().detach()
    g.readonly(True)
    nodes_train = np.arange(features.shape[0])[train_mask]
    nodes_test = np.arange(features.shape[0])[test_mask]

    y_bars = []
    for i in range(args.n_epochs):
        loss = train_on_fullgraph(g, nodes_train, fullgraph_steady_state_operator,
            predictor, optimizer, args.n_embedding_updates, args.n_parameter_updates, args.alpha)
        accuracy_train, _ = test(g, nodes_train, predictor)
        accuracy_test, z = test(g, nodes_test, predictor)
        print("Iter {:05d} | Train acc {:.4f} | Test acc {:.4f}".format(i, accuracy_train, accuracy_test))
        y_bar = F.log_softmax(z, 1)
        y_bars.append(y_bar)


if __name__ == '__main__':

    # get parameters
    parser = argparse.ArgumentParser(description="Try to find fixpoint")

    parser.add_argument('--dataset', default='cora', help='choose dataset from: cora, pubmed, citeseer')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_hidden', type=int, default=16)
    parser.add_argument('--n_input', type=int, default=2882)
    parser.add_argument('--n_output', type=int, default=7)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--n_embedding_updates', type=int, default=8)
    parser.add_argument('--n_parameter_updates', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()

    print(args)
    main(args)
    print("Finish!")