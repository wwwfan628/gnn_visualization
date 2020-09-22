from dgl.data import load_data
from dgl.data import PPIDataset
from dgl.data import TUDataset
from dgl.data.gnn_benckmark import Coauthor, AmazonCoBuy
import dgl
from dgl import DGLGraph
import numpy as np
import networkx as nx
import torch
from torch.utils.data import DataLoader
import yaml
import os

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

def load_reddit(args):
    data = load_data(args)
    features = torch.FloatTensor(data.features).to(device)
    labels = torch.LongTensor(data.labels).to(device)
    train_mask = torch.BoolTensor(data.train_mask).to(device)
    test_mask = torch.BoolTensor(data.test_mask).to(device)
    g = data.graph
    return g, features, labels, train_mask, test_mask

def collate_ppi(sample):
    graphs, labels =map(list, zip(*sample))
    graph = dgl.batch(graphs)
    labels = torch.from_numpy(np.concatenate(labels)).to(device)
    return graph, labels

def load_ppi(batch_size):
    train_dataset = PPIDataset(mode='train')
    valid_dataset = PPIDataset(mode='valid')
    test_dataset = PPIDataset(mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_ppi)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_ppi)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_ppi)
    return train_dataset, valid_dataset, test_dataset, train_dataloader, valid_dataloader, test_dataloader

def collate_tu(batch):
    graphs, labels = map(list, zip(*batch))
    # batch graphs and cast to PyTorch tensor
    for graph in graphs:
        for (key, value) in graph.ndata.items():
            graph.ndata[key] = torch.FloatTensor(value.float())
        for (key, value) in graph.edata.items():
            graph.edata[key] = torch.FloatTensor(value.float())
    batched_graphs = dgl.batch(graphs)
    # cast to PyTorch tensor
    batched_labels = torch.LongTensor(np.array(labels).flatten()).to(device)
    return batched_graphs, batched_labels


def load_tu(dataset_name, train_ratio, validate_ratio, batch_size):
    dataset = TUDataset(name=dataset_name)

    # Use provided node feature by default. If no feature provided, use node label instead.
    # If neither labels provided, use degree for node feature.
    for g in dataset.graph_lists:
        if 'node_attr' in g.ndata.keys():
            g.ndata['feat'] = g.ndata['node_attr']
        elif 'node_labels' in g.ndata.keys():
            g.ndata['feat'] = g.ndata['node_labels']
        else:
            g.ndata['feat'] = g.in_degrees().view(-1, 1).float()

    if 'node_attr' in g.ndata.keys():
        print("Use default node feature!")
    elif 'node_labels' in g.ndata.keys():
        print("Use node labels as node features!")
    else:
        print("Use node degree as node features!")

    statistics = dataset.statistics()

    train_size = int(train_ratio * len(dataset))
    valid_size = int(validate_ratio * len(dataset))
    test_size = int(len(dataset) - train_size - valid_size)

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_size, valid_size, test_size))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_tu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_tu)
    return statistics, train_dataset, valid_dataset, train_dataloader, valid_dataloader

def load_dataset(args):

    if args.dataset in 'cora, citeseer, pubmed':
        g, features, labels, train_mask, valid_mask, test_mask = load_citation(args)
        return g, features, labels, train_mask, valid_mask, test_mask

    elif args.dataset == 'reddit-self-loop':
        g, features, labels, train_mask, test_mask = load_reddit(args)
        return g, features, labels, train_mask, test_mask

    elif args.dataset in 'amazon_photo, amazon_computers, coauthors_cs, coauthors_physics':
        g, features, labels, train_mask, valid_mask, test_mask = load_amz_coauthors(args)
        return g, features, labels, train_mask, valid_mask, test_mask

    elif args.dataset == 'ppi':

        config_file = os.path.join(os.getcwd(), '../configs/ppi.yaml')
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        batch_size = config['batch_size']

        train_dataset, valid_dataset, test_dataset, train_dataloader, valid_dataloader, test_dataloader = load_ppi(batch_size)
        return train_dataset, valid_dataset, test_dataset, train_dataloader, valid_dataloader, test_dataloader

    elif 'tu' in args.dataset:

        path = '../configs/' + args.dataset + '.yaml'
        config_file = os.path.join(os.getcwd(), path)
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        dataset_name = config['dataset_name']
        train_ratio = config['train_ratio']
        validate_ratio = config['validate_ratio']
        batch_size = config['batch_size']

        statistics, train_dataset, valid_dataset, train_dataloader, valid_dataloader = load_tu(dataset_name, train_ratio, validate_ratio, batch_size)

        return statistics, train_dataset, valid_dataset, train_dataloader, valid_dataloader


