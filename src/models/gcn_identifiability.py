from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, num_layers, in_feats, h_feats, out_feats):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        if self.num_layers >= 2:
            # input layer
            self.gcn_layers.append(GraphConv(in_feats, h_feats))
            # intermediate layers
            for l in range(1, self.num_layers-1):
                self.gcn_layers.append(GraphConv(h_feats, h_feats))
            # output layer
            self.gcn_layers.append(GraphConv(h_feats, out_feats))
        else:
            self.gcn_layers.append(GraphConv(in_feats, out_feats))

    def forward(self, graph, inputs):
        outputs = []
        h = inputs
        outputs.append(h)
        for l in range(self.num_layers-1):
            h = F.relu(self.gcn_layers[l](graph, h))
            outputs.append(h)
        h = self.gcn_layers[self.num_layers-1](graph,h)
        outputs.append(h)
        return outputs