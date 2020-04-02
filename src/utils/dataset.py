from dgl.data import citation_graph
from dgl.data import RedditDataset
import networkx as nx
import torch
from dgl import DGLGraph


def load_cora_data():
    data = citation_graph.load_cora()
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


def load_reddit_data():
    data = RedditDataset(self_loop=True)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    g = data.graph
    return g, features, labels, train_mask, test_mask
