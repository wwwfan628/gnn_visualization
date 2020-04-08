from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, num_feats):
        super(GCN, self).__init__()
        self.gcn_layer1 = GraphConv(num_feats, num_feats)
        self.gcn_layer2 = GraphConv(num_feats, num_feats)
        self.gcn_layer3 = GraphConv(num_feats, num_feats)

    def forward(self, graph, inputs):
        h1 = F.relu(self.gcn_layer1(graph, inputs))
        h2 = F.relu(self.gcn_layer2(graph, h1))
        h3 = F.relu(self.gcn_layer3(graph, h2))
        return h3
