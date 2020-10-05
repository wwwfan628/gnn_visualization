import torch.nn.functional as F
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_layers, in_feats,h_feats,out_feats):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.fc_layers = nn.ModuleList()
        if self.num_layers != 1:
            # input layer
            self.fc_layers.append(nn.Linear(in_feats, h_feats))
            # intermediate layers
            for l in range(1, self.num_layers-1):
                self.fc_layers.append(nn.Linear(h_feats, h_feats))
            # output layer
            self.fc_layers.append(nn.Linear(h_feats, out_feats))
        else:
            self.fc_layers.append(nn.Linear(in_feats, out_feats))
    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers-1):
            h = F.relu(self.fc_layers[l](h))
        h = self.fc_layers[self.num_layers-1](h)
        return h