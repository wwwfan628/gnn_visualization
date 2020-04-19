from dgl.data import load_data
from dgl.data import LegacyPPIDataset
from dgl.data import LegacyTUDataset
import dgl
from dgl import DGLGraph
import numpy as np
import networkx as nx
import torch
from torch.utils.data import DataLoader
import yaml
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def load_reddit(args):
    data = load_data(args)
    features = torch.FloatTensor(data.features).to(device)
    labels = torch.LongTensor(data.labels).to(device)
    train_mask = torch.BoolTensor(data.train_mask).to(device)
    test_mask = torch.BoolTensor(data.test_mask).to(device)
    g = data.graph
    return g, features, labels, train_mask, test_mask

def collate_ppi(sample):
    graphs, feats, labels =map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats)).to(device)
    labels = torch.from_numpy(np.concatenate(labels)).to(device)
    return graph, feats, labels

def load_ppi(batch_size):
    train_dataset = LegacyPPIDataset(mode='train')
    valid_dataset = LegacyPPIDataset(mode='valid')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_ppi)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_ppi)
    return train_dataset, train_dataloader, valid_dataloader

def collate_tu(batch):
    graphs, labels = map(list, zip(*batch))
    # batch graphs and cast to PyTorch tensor
    for graph in graphs:
        for (key, value) in graph.ndata.items():
            graph.ndata[key] = torch.FloatTensor(value.float()).to(device)
    batched_graphs = dgl.batch(graphs)
    # cast to PyTorch tensor
    batched_labels = torch.LongTensor(np.array(labels)).to(device)
    return batched_graphs, batched_labels

def load_tu(dataset_name, train_ratio, validate_ratio, batch_size):
    dataset = LegacyTUDataset(name=dataset_name)
    statistics = dataset.statistics()
    train_size = int(train_ratio * len(dataset))
    valid_size = int(validate_ratio * len(dataset))
    test_size = int(len(dataset) - train_size - valid_size)

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_size, valid_size, test_size))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_tu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_tu)
    return statistics, train_dataset, train_dataloader, valid_dataloader

def load_dataset(args):

    if args.dataset in 'cora, citeseer, pubmed':
        g, features, labels, train_mask, test_mask = load_citation(args)
        return g, features, labels, train_mask, test_mask

    elif args.dataset == 'reddit-self-loop':
        g, features, labels, train_mask, test_mask = load_reddit(args)
        return g, features, labels, train_mask, test_mask

    elif args.dataset == 'ppi':

        config_file = os.path.join(os.getcwd(), '../configs/ppi.yaml')
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        batch_size = config['batch_size']

        train_dataset, train_dataloader, valid_dataloader = load_ppi(batch_size)
        return train_dataset, train_dataloader, valid_dataloader

    elif args.dataset in 'aids, imdb-binary, reddit-binary':

        path = '../configs/' + args.dataset + '.yaml'
        config_file = os.path.join(os.getcwd(), path)
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        dataset_name = config['dataset_name']
        train_ratio = config['train_ratio']
        validate_ratio = config['validate_ratio']
        batch_size = config['batch_size']

        statistics, train_dataset, train_dataloader, valid_dataloader = load_tu(dataset_name, train_ratio, validate_ratio, batch_size)
        return statistics, train_dataset, train_dataloader, valid_dataloader


