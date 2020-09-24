from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GCN, self).__init__()
        self.gcn_layer_1 = GraphConv(in_feats, h_feats)
        self.gcn_layer_2 = GraphConv(h_feats, h_feats)
        self.gcn_layer_3 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs):
        h1 = F.relu(self.gcn_layer_1(graph,inputs))
        h2 = F.relu(self.gcn_layer_2(graph, h1))
        h3 = F.relu(self.gcn_layer_2(graph, h2))
        h4 = self.gcn_layer_3(graph,h2)
        return h4, h3, h2