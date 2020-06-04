from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import torch.nn as nn


class GCN_Baseline(nn.Module):
    def __init__(self,in_feats,h_feats,out_feats):
        super(GCN_Baseline, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs):
        h1 = F.relu(self.gcn_layer1(graph, inputs))
        h2 = self.gcn_layer2(graph, h1)
        return h2, h1, inputs

class GCN_Baseline_3Layers(nn.Module):
    def __init__(self,in_feats,h_feats,out_feats):
        super(GCN_Baseline_3Layers, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, h_feats)
        self.gcn_layer3 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs):
        h1 = F.relu(self.gcn_layer1(graph, inputs))
        h2 = F.relu(self.gcn_layer2(graph, h1))
        h3 = self.gcn_layer3(graph,h2)
        return h3, h2, h1, inputs


