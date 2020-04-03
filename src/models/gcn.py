from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_feats, h1_feats, h2_feats, num_classes):
        super(GCN, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h1_feats)
        self.gcn_layer2 = GraphConv(h1_feats, h2_feats)
        self.gcn_layer3 = GraphConv(h2_feats, num_classes)

    def forward(self, graph, inputs):
        h1 = self.gcn_layer1(graph, inputs)
        h1 = F.relu(h1)
        h2 = self.gcn_layer2(graph, h1)
        h2 = F.relu(h2)
        h3 = self.gcn_layer3(graph, h2)
        return h3