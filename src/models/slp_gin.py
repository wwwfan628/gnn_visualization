from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling
import torch.nn.functional as F
import torch.nn as nn

class SLP_GIN_4node(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(SLP_GIN_4node, self).__init__()
        self.gin_layers = nn.ModuleList()
        self.fc = nn.Linear(in_feats, h_feats)
        for i in range(3):
            self.gin_layers.append(GINConv(nn.Sequential(nn.Dropout(0.6), nn.Linear(h_feats, h_feats), nn.ReLU()),
                                           'mean', 0, True))
        self.last_layer = GINConv(nn.Sequential(nn.Dropout(0.6), nn.Linear(h_feats, out_feats)),
                                  'mean', 0, True)

    def forward(self, graph, inputs):
        h = F.relu(self.fc(inputs))
        for layer in self.gin_layers:
            h = layer(graph, h)
        h = self.last_layer4(graph, h)
        return h


class SLP_GIN_4graph(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(SLP_GIN_4graph, self).__init__()
        self.gin_layers = nn.ModuleList()
        self.drop = nn.Dropout(0.6)
        self.pool = SumPooling()

        self.fc = nn.Linear(in_feats, h_feats)
        for i in range(3):
            self.gin_layers.append(GINConv(nn.Sequential(nn.Dropout(0.6), nn.Linear(h_feats, h_feats), nn.ReLU()),
                                           'mean', 0, True))
        # classification
        self.last_layer = nn.Linear(h_feats, out_feats)

    def forward(self, graph, inputs):
        h = F.relu(self.fc(inputs))
        hidden_rep = [h]
        for layer in self.gin_layers:
            h = layer(graph, h)
            hidden_rep.append(h)

        # read-out function
        score_over_layer = 0
        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(graph, h)
            score_over_layer += self.drop(self.last_layer(pooled_h))

        return score_over_layer
