from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import torch.nn as nn
import dgl


class SLP_GCN_4node(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(SLP_GCN_4node, self).__init__()
        self.fc = nn.Linear(in_feats, h_feats)
        self.gcn_layer1 = GraphConv(h_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, h_feats)
        self.gcn_layer3 = GraphConv(h_feats, h_feats)
        self.gcn_layer4 = GraphConv(h_feats, out_feats)

    def forward(self, graph, inputs):
        h1 = F.relu(self.fc(inputs))
        h2 = F.relu(self.gcn_layer1(graph, h1))
        h3 = F.relu(self.gcn_layer2(graph, h2))
        h4 = F.relu(self.gcn_layer3(graph, h3))
        h5 = self.gcn_layer4(graph, h4)
        return h5


class SLP_GCN_4graph(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(SLP_GCN_4graph, self).__init__()
        self.fc = nn.Linear(in_feats, h_feats)
        self.gcn_layer1 = GraphConv(h_feats, h_feats)
        self.gcn_layer2 = GraphConv(h_feats, h_feats)
        self.gcn_layer3 = GraphConv(h_feats, h_feats)
        # classification
        self.classify = nn.Linear(h_feats, out_feats)

    def forward(self, graph, inputs):
        h1 = F.relu(self.fc(inputs))
        h2 = F.relu(self.gcn_layer1(graph, h1))
        h3 = F.relu(self.gcn_layer2(graph, h2))
        h4 = F.relu(self.gcn_layer3(graph, h3))

        graph.ndata['h'] = h4
        # read-out function
        graph_repr = dgl.mean_nodes(graph, 'h')
        h5 = self.classify(graph_repr)
        return h5
