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

def load_citation(dataset):
    data = load_data(dataset)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, train_mask, test_mask

def load_reddit():
    data = load_data('reddit-self-loop')
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    g = data.graph
    return g, features, labels, train_mask, test_mask

def collate_ppi(sample):
    graphs, feats, labels =map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
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
            graph.ndata[key] = torch.FloatTensor(value.float())
    batched_graphs = dgl.batch(graphs)
    # cast to PyTorch tensor
    batched_labels = torch.LongTensor(np.array(labels))
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

def load_dataset(dataset):

    if dataset == 'cora' or 'citeseer' or 'pubmed':
        g, features, labels, train_mask, test_mask = load_citation(dataset)
        return g, features, labels, train_mask, test_mask

    elif dataset == 'reddit':
        g, features, labels, train_mask, test_mask = load_reddit()
        return g, features, labels, train_mask, test_mask

    elif dataset == 'ppi':

        config_file = os.path.join(os.getcwd(), '../configs/ppi.yaml')
        with open(config_file, 'r') as f:
            config = yaml.load(f)
        batch_size = config['batch_size']

        train_dataset, train_dataloader, valid_dataloader = load_ppi(batch_size)
        return train_dataset, train_dataloader, valid_dataloader

    elif dataset == 'tu':

        config_file = os.path.join(os.getcwd(), '../configs/tu.yaml')
        with open(config_file, 'r') as f:
            config = yaml.load(f)
        dataset_name = config['dataset_name']
        train_ratio = config['train_ratio']
        validate_ratio = config['validate_ratio']
        batch_size = config['batch_size']

        statistics, train_dataloader, valid_dataloader = load_tu(dataset_name, train_ratio, validate_ratio, batch_size)
        return statistics, train_dataloader, valid_dataloader


