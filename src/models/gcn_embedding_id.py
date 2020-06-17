from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import torch.nn as nn


class GCN_2Layers(nn.Module):
    def __init__(self,in_feats,h_feats,out_feats):
        super(GCN_2Layers, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs, args):
        h1 = F.relu(self.gcn_layer1(graph, inputs))
        h2 = self.gcn_layer2(graph, h1)
        if args.embedding_layer == '1':
            return h2, h1, inputs
        elif args.embedding_layer == '2':
            return h2, h2, inputs


class GCN_3Layers(nn.Module):
    def __init__(self,in_feats,h_feats,out_feats):
        super(GCN_3Layers, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, h_feats)
        self.gcn_layer3 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs, args):
        h1 = F.relu(self.gcn_layer1(graph, inputs))
        h2 = F.relu(self.gcn_layer2(graph, h1))
        h3 = self.gcn_layer3(graph,h2)
        if args.embedding_layer == '1':
            return h3, h1, inputs
        elif args.embedding_layer == '2':
            return h3, h2, inputs
        elif args.embedding_layer == '3':
            return h3, h3, inputs


class GCN_4Layers(nn.Module):
    def __init__(self,in_feats,h_feats,out_feats):
        super(GCN_4Layers, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, h_feats)
        self.gcn_layer3 = GraphConv(h_feats, h_feats)
        self.gcn_layer4 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs, args):
        h1 = F.relu(self.gcn_layer1(graph, inputs))
        h2 = F.relu(self.gcn_layer2(graph, h1))
        h3 = F.relu(self.gcn_layer3(graph, h2))
        h4 = self.gcn_layer4(graph, h3)
        if args.embedding_layer == '1':
            return h4, h1, inputs
        elif args.embedding_layer == '2':
            return h4, h2, inputs
        elif args.embedding_layer == '3':
            return h4, h3, inputs
        elif args.embedding_layer == '4':
            return h4, h4, inputs


class GCN_5Layers(nn.Module):
    def __init__(self,in_feats,h_feats,out_feats):
        super(GCN_5Layers, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, h_feats)
        self.gcn_layer3 = GraphConv(h_feats, h_feats)
        self.gcn_layer4 = GraphConv(h_feats, h_feats)
        self.gcn_layer5 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs, args):
        h1 = F.relu(self.gcn_layer1(graph, inputs))
        h2 = F.relu(self.gcn_layer2(graph, h1))
        h3 = F.relu(self.gcn_layer3(graph, h2))
        h4 = F.relu(self.gcn_layer4(graph, h3))
        h5 = self.gcn_layer5(graph, h4)
        if args.embedding_layer == '1':
            return h5, h1, inputs
        elif args.embedding_layer == '2':
            return h5, h2, inputs
        elif args.embedding_layer == '3':
            return h5, h3, inputs
        elif args.embedding_layer == '4':
            return h5, h4, inputs
        elif args.embedding_layer == '5':
            return h5, h5, inputs


class GCN_6Layers(nn.Module):
    def __init__(self,in_feats,h_feats,out_feats):
        super(GCN_6Layers, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, h_feats)
        self.gcn_layer3 = GraphConv(h_feats, h_feats)
        self.gcn_layer4 = GraphConv(h_feats, h_feats)
        self.gcn_layer5 = GraphConv(h_feats, h_feats)
        self.gcn_layer6 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs, args):
        h1 = F.relu(self.gcn_layer1(graph, inputs))
        h2 = F.relu(self.gcn_layer2(graph, h1))
        h3 = F.relu(self.gcn_layer3(graph, h2))
        h4 = F.relu(self.gcn_layer4(graph, h3))
        h5 = F.relu(self.gcn_layer5(graph, h4))
        h6 = self.gcn_layer5(graph, h5)
        if args.embedding_layer == '1':
            return h6, h1, inputs
        elif args.embedding_layer == '2':
            return h6, h2, inputs
        elif args.embedding_layer == '3':
            return h6, h3, inputs
        elif args.embedding_layer == '4':
            return h6, h4, inputs
        elif args.embedding_layer == '5':
            return h6, h5, inputs
        elif args.embedding_layer == '6':
            return h6, h6, inputs