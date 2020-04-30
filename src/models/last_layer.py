from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import torch.nn as nn
import dgl


class Last_Layer_4node(nn.Module):
    def __init__(self, h_feats, out_feats):
        super(Last_Layer_4node, self).__init__()
        self.gcn_layer4 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs):
        h = self.gcn_layer4(graph, inputs)
        return h


class Last_Layer_4graph(nn.Module):
    def __init__(self, h_feats, out_feats):
        super(Last_Layer_4graph, self).__init__()
        # classification
        self.classify = nn.Linear(h_feats, out_feats)

    def forward(self, graph, inputs):
        graph.ndata['h'] = inputs
        # read-out function
        graph_repr = dgl.mean_nodes(graph, 'h')
        h = self.classify(graph_repr)
        return h
