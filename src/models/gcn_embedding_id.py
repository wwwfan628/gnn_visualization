from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import torch.nn as nn


class GCN_1Layer(nn.Module):
    def __init__(self,in_feats,out_feats):
        super(GCN_1Layer, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, out_feats)

    def forward(self, graph, inputs):
        h1 = self.gcn_layer1(graph, inputs)
        return h1, h1

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
        h6 = self.gcn_layer6(graph, h5)
        return h6, h5, h4, h3, h2, h1


class GCN_7Layers(nn.Module):
    def __init__(self,in_feats,h_feats,out_feats):
        super(GCN_7Layers, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, h_feats)
        self.gcn_layer3 = GraphConv(h_feats, h_feats)
        self.gcn_layer4 = GraphConv(h_feats, h_feats)
        self.gcn_layer5 = GraphConv(h_feats, h_feats)
        self.gcn_layer6 = GraphConv(h_feats, h_feats)
        self.gcn_layer7 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs):
        h1 = F.relu(self.gcn_layer1(graph, inputs))
        h2 = F.relu(self.gcn_layer2(graph, h1))
        h3 = F.relu(self.gcn_layer3(graph, h2))
        h4 = F.relu(self.gcn_layer4(graph, h3))
        h5 = F.relu(self.gcn_layer5(graph, h4))
        h6 = F.relu(self.gcn_layer6(graph, h5))
        h7 = self.gcn_layer7(graph, h6)
        return h7, h6, h5, h4, h3, h2, h1


class GCN_8Layers(nn.Module):
    def __init__(self,in_feats,h_feats,out_feats):
        super(GCN_8Layers, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, h_feats)
        self.gcn_layer3 = GraphConv(h_feats, h_feats)
        self.gcn_layer4 = GraphConv(h_feats, h_feats)
        self.gcn_layer5 = GraphConv(h_feats, h_feats)
        self.gcn_layer6 = GraphConv(h_feats, h_feats)
        self.gcn_layer7 = GraphConv(h_feats, h_feats)
        self.gcn_layer8 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs):
        h1 = F.relu(self.gcn_layer1(graph, inputs))
        h2 = F.relu(self.gcn_layer2(graph, h1))
        h3 = F.relu(self.gcn_layer3(graph, h2))
        h4 = F.relu(self.gcn_layer4(graph, h3))
        h5 = F.relu(self.gcn_layer5(graph, h4))
        h6 = F.relu(self.gcn_layer6(graph, h5))
        h7 = F.relu(self.gcn_layer7(graph, h6))
        h8 = self.gcn_layer8(graph, h7)
        return h8, h7, h6, h5, h4, h3, h2, h1


class GCN_9Layers(nn.Module):
    def __init__(self,in_feats,h_feats,out_feats):
        super(GCN_9Layers, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, h_feats)
        self.gcn_layer3 = GraphConv(h_feats, h_feats)
        self.gcn_layer4 = GraphConv(h_feats, h_feats)
        self.gcn_layer5 = GraphConv(h_feats, h_feats)
        self.gcn_layer6 = GraphConv(h_feats, h_feats)
        self.gcn_layer7 = GraphConv(h_feats, h_feats)
        self.gcn_layer8 = GraphConv(h_feats, h_feats)
        self.gcn_layer9 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs):
        h1 = F.relu(self.gcn_layer1(graph, inputs))
        h2 = F.relu(self.gcn_layer2(graph, h1))
        h3 = F.relu(self.gcn_layer3(graph, h2))
        h4 = F.relu(self.gcn_layer4(graph, h3))
        h5 = F.relu(self.gcn_layer5(graph, h4))
        h6 = F.relu(self.gcn_layer6(graph, h5))
        h7 = F.relu(self.gcn_layer7(graph, h6))
        h8 = F.relu(self.gcn_layer8(graph, h7))
        h9 = self.gcn_layer9(graph, h8)
        return h9, h8, h7, h6, h5, h4, h3, h2, h1


class GCN_10Layers(nn.Module):
    def __init__(self,in_feats,h_feats,out_feats):
        super(GCN_10Layers, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, h_feats)
        self.gcn_layer3 = GraphConv(h_feats, h_feats)
        self.gcn_layer4 = GraphConv(h_feats, h_feats)
        self.gcn_layer5 = GraphConv(h_feats, h_feats)
        self.gcn_layer6 = GraphConv(h_feats, h_feats)
        self.gcn_layer7 = GraphConv(h_feats, h_feats)
        self.gcn_layer8 = GraphConv(h_feats, h_feats)
        self.gcn_layer9 = GraphConv(h_feats, h_feats)
        self.gcn_layer10 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs):
        h1 = F.relu(self.gcn_layer1(graph, inputs))
        h2 = F.relu(self.gcn_layer2(graph, h1))
        h3 = F.relu(self.gcn_layer3(graph, h2))
        h4 = F.relu(self.gcn_layer4(graph, h3))
        h5 = F.relu(self.gcn_layer5(graph, h4))
        h6 = F.relu(self.gcn_layer6(graph, h5))
        h7 = F.relu(self.gcn_layer7(graph, h6))
        h8 = F.relu(self.gcn_layer8(graph, h7))
        h9 = F.relu(self.gcn_layer9(graph, h8))
        h10 = self.gcn_layer10(graph, h9)
        return h10, h9, h8, h7, h6, h5, h4, h3, h2, h1