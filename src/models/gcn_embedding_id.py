from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import torch.nn as nn


class GCN_2Layers(nn.Module):
    def __init__(self,in_feats,h_feats,out_feats):
        super(GCN_2Layers, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs):
        h1 = F.relu(self.gcn_layer1(graph, inputs))
        h2 = self.gcn_layer2(graph, h1)
        return h2, h1


class GCN_3Layers(nn.Module):
    def __init__(self,in_feats,h_feats,out_feats):
        super(GCN_3Layers, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, h_feats)
        self.gcn_layer3 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs):
        h1 = F.relu(self.gcn_layer1(graph, inputs))
        h2 = F.relu(self.gcn_layer2(graph, h1))
        h3 = self.gcn_layer3(graph,h2)
        return h3, h2, h1


class GCN_4Layers(nn.Module):
    def __init__(self,in_feats,h_feats,out_feats):
        super(GCN_4Layers, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, h_feats)
        self.gcn_layer3 = GraphConv(h_feats, h_feats)
        self.gcn_layer4 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs):
        h1 = F.relu(self.gcn_layer1(graph, inputs))
        h2 = F.relu(self.gcn_layer2(graph, h1))
        h3 = F.relu(self.gcn_layer3(graph, h2))
        h4 = self.gcn_layer4(graph, h3)
        return h4, h3, h2, h1


class GCN_5Layers(nn.Module):
    def __init__(self,in_feats,h_feats,out_feats):
        super(GCN_5Layers, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, h_feats)
        self.gcn_layer3 = GraphConv(h_feats, h_feats)
        self.gcn_layer4 = GraphConv(h_feats, h_feats)
        self.gcn_layer5 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs):
        h1 = F.relu(self.gcn_layer1(graph, inputs))
        h2 = F.relu(self.gcn_layer2(graph, h1))
        h3 = F.relu(self.gcn_layer3(graph, h2))
        h4 = F.relu(self.gcn_layer4(graph, h3))
        h5 = self.gcn_layer5(graph, h4)
        return h5, h4, h3, h2, h1


class GCN_6Layers(nn.Module):
    def __init__(self,in_feats,h_feats,out_feats):
        super(GCN_6Layers, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, h_feats)
        self.gcn_layer3 = GraphConv(h_feats, h_feats)
        self.gcn_layer4 = GraphConv(h_feats, h_feats)
        self.gcn_layer5 = GraphConv(h_feats, h_feats)
        self.gcn_layer6 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs):
        h1 = F.relu(self.gcn_layer1(graph, inputs))
        h2 = F.relu(self.gcn_layer2(graph, h1))
        h3 = F.relu(self.gcn_layer3(graph, h2))
        h4 = F.relu(self.gcn_layer4(graph, h3))
        h5 = F.relu(self.gcn_layer5(graph, h4))
        h6 = self.gcn_layer5(graph, h5)
        return h6, h5, h4, h3, h2, h1