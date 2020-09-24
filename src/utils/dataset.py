from dgl.data import load_data
from dgl.data.gnn_benckmark import Coauthor, AmazonCoBuy
from dgl import DGLGraph
import numpy as np
import networkx as nx
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('dataset.py running on {}!'.format(device))

def load_citation(args):
    data = load_data(args)
    features = torch.FloatTensor(data.features).to(device)
    labels = torch.LongTensor(data.labels).to(device)
    train_mask = torch.BoolTensor(data.train_mask).to(device)
    valid_mask = torch.BoolTensor(data.val_mask).to(device)
    test_mask = torch.BoolTensor(data.test_mask).to(device)
    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, train_mask, valid_mask, test_mask

def index_to_mask(index_set, size):
    mask = torch.zeros(size, dtype=torch.bool)
    index = np.array(list(index_set))
    mask[index] = 1
    return mask

def load_amz_coauthors(args):
    # Set random coauthor/co-purchase splits:
    # 20 per class for training
    # 30 per classes for validation
    # rest labels for testing
    if 'amazon' in args.dataset:
        name = args.dataset.split('_')[-1]
        dataset = AmazonCoBuy(name)
    elif 'coauthors' in args.dataset:
        name = args.dataset.split('_')[-1]
        dataset = Coauthor(name)
    g = dataset.data[0]
    features = torch.FloatTensor(g.ndata['feat'])
    labels = torch.LongTensor(g.ndata['label'])

    indices = []
    num_classes = torch.max(labels).item() + 1
    for i in range(num_classes):
        index = (labels == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)

    test_index = torch.cat([i[50:] for i in indices], dim=0)
    test_index = test_index[torch.randperm(test_index.size(0))]

    train_mask = index_to_mask(train_index, size=g.number_of_nodes())
    val_mask = index_to_mask(val_index, size=g.number_of_nodes())
    test_mask = index_to_mask(test_index, size=g.number_of_nodes())
    return g, features, labels, train_mask, val_mask, test_mask


def load_dataset(args):

    if args.dataset in 'cora, citeseer, pubmed':
        g, features, labels, train_mask, valid_mask, test_mask = load_citation(args)
        return g, features, labels, train_mask, valid_mask, test_mask


    elif args.dataset in 'amazon_photo, amazon_computers, coauthors_cs, coauthors_physics':
        g, features, labels, train_mask, valid_mask, test_mask = load_amz_coauthors(args)
        return g, features, labels, train_mask, valid_mask, test_mask


