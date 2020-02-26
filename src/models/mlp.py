import torch.nn.functional as F
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_feats, h1_feats, h2_feats, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_feats, h1_feats)
        self.fc2 = nn.Linear(h1_feats, h2_feats)
        self.fc3 = nn.Linear(h2_feats, num_classes)

    def forward(self, inputs):
        h1 = F.relu(self.fc1(inputs))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return h3
