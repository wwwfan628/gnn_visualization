import torch
import time
import numpy as np
import torch.nn.functional as F
import yaml
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def modify_features(num_nodes, feat_size):
    new_features = torch.rand([num_nodes, feat_size])
    new_features = (new_features.t() / torch.sum(new_features, dim=1)).t()  # normalize sum of each row to 1
    return new_features


def evaluate_gcn(model, graph, features, labels, mask):
    # used to evaluate the performance of the model on test dataset
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logp = F.log_softmax(logits, 1)
        loss_test = F.nll_loss(logp[mask], labels[mask])
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices[mask] == labels[mask])
        acc = correct.item() * 1.0 / len(labels[mask])
    return acc, loss_test


def evaluate_and_classify_nodes_gcn(model, graph, features, labels, mask):
    # evaluate classification accuracy on test dataset and return correctly classified nodes
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices[mask] == labels[mask])
        acc = correct.item() * 1.0 / len(labels[mask])
        correctly_classified_nodes = set(np.arange(graph.number_of_nodes())[indices == labels])
    return acc, correctly_classified_nodes


def evaluate_and_classify_nodes_with_random_features_gcn(model, graph, features, labels, mask):
    # generate random features
    random_features = modify_features(features.shape[0], features.shape[1])
    model.eval()
    with torch.no_grad():
        logits = model(graph, random_features)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices[mask] == labels[mask])
        acc = correct.item() * 1.0 / len(labels[mask])
        correctly_classified_nodes = set(np.arange(graph.number_of_nodes())[indices == labels])

    return acc, correctly_classified_nodes


def train_gcn(net, graph, features, labels, train_mask, valid_mask, args):
    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    lr = config['train_lr']  # learning rate
    max_epoch = config['train_max_epoch']  # maximal number of training epochs
    # used for early stop
    patience = config['train_patience']
    best_acc = -1
    best_loss = float('inf')
    cur_step = 0

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    dur = []

    for epoch in range(max_epoch):
        t0 = time.time()

        net.train()
        logits = net(graph, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dur.append(time.time() - t0)
        acc, loss_valid = evaluate_gcn(net, graph, features, labels, valid_mask)
        print("Epoch {:04d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(epoch+1, loss.item(), acc, np.mean(dur)))

        # early stop
        if acc > best_acc or best_loss > loss_valid:
            best_acc = np.max((acc, best_acc))
            best_loss = np.min((best_loss, loss_valid))
            cur_step = 0
        else:
            cur_step += 1
            if cur_step == patience:
                break
    return best_acc


def evaluate_mlp(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss_test = F.nll_loss(logp[mask], labels[mask])
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices[mask] == labels[mask])
        acc = correct.item() * 1.0 / len(labels)
    return acc, loss_test


def evaluate_and_classify_nodes_mlp(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices[mask] == labels[mask])
        acc = correct.item() * 1.0 / len(labels[mask])
        correctly_classified_nodes = set(np.arange(features.shape[0])[indices == labels])
    return acc, correctly_classified_nodes


def train_mlp(model, features, labels, train_mask, valid_mask, args):
    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    lr = config['train_lr']  # learning rate
    max_epoch = config['train_max_epoch']  # maximal number of training epochs
    # used for early stop
    patience = config['train_patience']
    best_acc = -1
    best_loss = float('inf')
    cur_step = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dur = []

    for epoch in range(max_epoch):
        t0 = time.time()

        model.train()
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dur.append(time.time() - t0)
        acc, loss_valid = evaluate_mlp(model, features, labels, valid_mask)
        print("Epoch {:04d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(epoch + 1, loss.item(), acc, np.mean(dur)))

        # early stop
        if acc > best_acc or best_loss > loss_valid:
            best_acc = np.max((acc, best_acc))
            best_loss = np.min((best_loss, loss_valid))
            cur_step = 0
        else:
            cur_step += 1
            if cur_step == patience:
                break
    return best_acc






