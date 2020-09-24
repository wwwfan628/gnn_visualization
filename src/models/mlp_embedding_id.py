import torch.nn.functional as F
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(in_feats, h_feats)
        self.fc_2 = nn.Linear(h_feats, h_feats)
        self.fc_3 = nn.Linear(h_feats, out_feats)

    def forward(self, inputs):
        h1 = F.relu(self.fc_1(inputs))
        h2 = F.relu(self.fc_2(h1))
        h3 = self.fc_3(h2)
        return h3
