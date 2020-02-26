from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import torch.nn as nn


class MLP_GCN(nn.Module):
    def __init__(self, in_feats, h1_feats, h2_feats, h3_feats, h4_feats, h5_feats, h6_feats, num_classes):
        super(MLP_GCN, self).__init__()
        self.fc1 = nn.Linear(in_feats, h1_feats)
        self.fc2 = nn.Linear(h1_feats, h2_feats)
        self.fc3 = nn.Linear(h2_feats, h3_feats)
        self.gcn_layer1 = GraphConv(h3_feats, h4_feats)
        self.gcn_layer2 = GraphConv(h4_feats, h5_feats)
        self.gcn_layer3 = GraphConv(h5_feats, h6_feats)
        self.gcn_layer4 = GraphConv(h6_feats, num_classes)

    def forward(self, graph, inputs):
        h1 = F.relu(self.fc1(inputs))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = self.gcn_layer1(graph, h3)
        h4 = F.relu(h4)
        h5 = self.gcn_layer2(graph, h4)
        h5 = F.relu(h5)
        h6 = self.gcn_layer3(graph, h5)
        h6 = F.relu(h6)
        h7 = self.gcn_layer4(graph, h6)
        h7 = F.relu(h7)
        return h7
