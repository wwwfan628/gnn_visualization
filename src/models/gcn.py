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


class GCN_Baseline(nn.Module):
    def __init__(self,in_feats,h_feats,out_feats):
        super(GCN_Baseline, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs):
        h1 = F.relu(self.gcn_layer1(graph, inputs))
        h2 = self.gcn_layer2(graph, h1)
        return h2

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
        return h3


class GCN_SSE(nn.Module):
    def __init__(self,in_feats,h_feats,out_feats, alpha=0.1):
        super(GCN_SSE, self).__init__()
        self.fc = nn.Linear(in_feats, h_feats)
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, h_feats)
        self.gcn_layer3 = GraphConv(h_feats, out_feats)

        self.alpha = alpha

    def forward(self, graph, inputs):
        h1 = F.relu(self.gcn_layer1(graph, inputs))
        h2 = F.relu(self.gcn_layer2(graph, h1))
        h2_update = (1 - self.alpha) * h1 + self.alpha * h2
        h3 = self.gcn_layer3(graph,h2_update)
        return h3

