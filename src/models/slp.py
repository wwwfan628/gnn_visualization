import torch.nn.functional as F
import torch.nn as nn


class SLP(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(SLP, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats)

    def forward(self, inputs):
        h = F.relu(self.fc(inputs))
        return h
