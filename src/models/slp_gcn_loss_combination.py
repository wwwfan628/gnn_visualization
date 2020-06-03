from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import torch.nn as nn


class SLP_GCN_4node(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(SLP_GCN_4node, self).__init__()
        self.encoder = GraphConv(in_feats, h_feats)
        self.gcn_layer = GraphConv(h_feats, h_feats)
        self.decoder = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs):
        h1 = F.relu(self.encoder(graph,inputs))
        h2 = F.relu(self.gcn_layer(graph, h1))
        h3 = F.relu(self.gcn_layer(graph, h2))
        h4 = self.decoder(graph,h2)
        return h4, h3, h2
