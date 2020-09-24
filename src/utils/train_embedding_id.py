import torch
import time
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import yaml
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def classify_nodes(model, graph, features, labels):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)[-1]
        _, indices = torch.max(logits, dim=1)
        correctly_classified_nodes = set(np.arange(graph.number_of_nodes())[indices==labels])
        nodes = set(np.arange(graph.number_of_nodes()))
        incorrectly_classified_nodes = nodes.difference(correctly_classified_nodes)
    return correctly_classified_nodes, incorrectly_classified_nodes


def evaluate(model, graph, features, labels, mask):
    # used to evaluate the performance of the model on test dataset
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)[-1]
        logp = F.log_softmax(logits, 1)
        loss_test = F.nll_loss(logp[mask], labels[mask])
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        acc = correct.item() * 1.0 / len(labels)
        return acc, loss_test


def train_gcn(net, graph, features, labels, train_mask, test_mask, args):
    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    lr = config['train_lr']  # learning rate
    max_epoch = config['train_max_epoch']  # maximal number of training epochs
    # used for early stop
    patience = config['train_patience']
    best_accuracy = -1
    best_loss = float('inf')
    cur_step = 0

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    dur = []

    for epoch in range(max_epoch):
        t0 = time.time()

        net.train()
        logits = net(graph, features)[-1]
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # validation
        dur.append(time.time() - t0)
        acc, loss_test = evaluate(net, graph, features, labels, test_mask)
        print("Epoch {:04d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(epoch+1, loss.item(), acc, np.mean(dur)))

        # early stop
        if acc > best_accuracy or best_loss > loss_test:
            best_accuracy = np.max((acc, best_accuracy))
            best_loss = np.min((best_loss, loss_test))
            cur_step = 0
        else:
            cur_step += 1
            if cur_step == patience:
                break
    return best_accuracy


def evaluate_regression(model, inputs, targets, mask, loss_fcn):
    # used to evaluate the regression model
    model.eval()
    with torch.no_grad():
        predicted_outputs = model(inputs)
        y = torch.ones(targets[mask].shape[0]).to(device)
        loss_test = loss_fcn(predicted_outputs[mask], targets[mask], y)
        return loss_test


def train_regression(net, inputs, targets, train_mask, test_mask, args):
    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    lr = config['train_lr']  # learning rate
    max_epoch = config['train_max_epoch']  # maximal number of training epochs
    # used for early stop
    patience = config['train_patience']
    best_loss = float('inf')
    cur_step = 0

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fcn = nn.CosineEmbeddingLoss()
    dur = []

    for epoch in range(max_epoch):
        t0 = time.time()

        net.train()
        predicted_outputs = net(inputs)
        y = torch.ones(targets[train_mask].shape[0]).to(device)
        loss = loss_fcn(predicted_outputs[train_mask], targets[train_mask], y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        dur.append(time.time() - t0)
        loss_test = evaluate_regression(net, inputs, targets, test_mask, loss_fcn)
        print("Epoch {:04d} | Loss {:.4f} | Test Loss {:.4f} | Time(s) {:.4f}".format(epoch+1, loss.item(), loss_test, np.mean(dur)))

        # early stop
        if best_loss > loss_test:
            best_loss = np.min((best_loss, loss_test))
            cur_step = 0
        else:
            cur_step += 1
            if cur_step == patience:
                break
