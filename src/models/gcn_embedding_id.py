from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import torch.nn as nn


class GCN_Baseline(nn.Module):
    def __init__(self,in_feats,h_feats,out_feats):
        super(GCN_Baseline, self).__init__()
        self.gcn_layer1 = GraphConv(in_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs, args):
        h1 = F.relu(self.gcn_layer1(graph, inputs))
        h2 = self.gcn_layer2(graph, h1)
        if args.embedding_layer == '1':
            return h2, h1, inputs
        elif args.embedding_layer == '2':
            return h2, h2, inputs

class GCN_Baseline_3Layers(nn.Module):
    def __init__(self,in_feats,h_feats,out_feats):
        super(GCN_Baseline_3Layers, self).__init__()
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


