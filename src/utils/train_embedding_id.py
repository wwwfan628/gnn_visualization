import torch
import time
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import yaml
import os
from sklearn.metrics import f1_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def classify_nodes_citation(model, graph, features, labels):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)[0]
        _, indices = torch.max(logits, dim=1)
        correctly_classified_nodes = set(np.arange(graph.number_of_nodes())[indices==labels])
        nodes = set(np.arange(graph.number_of_nodes()))
        incorrectly_classified_nodes = nodes.difference(correctly_classified_nodes)
    return correctly_classified_nodes, incorrectly_classified_nodes


def evaluate_citation(model, graph, features, labels, mask):
    # used to evaluate the performance of the model on test dataset
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)[0]
        logp = F.log_softmax(logits, 1)
        loss_test = F.nll_loss(logp[mask], labels[mask])
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        acc = correct.item() * 1.0 / len(labels)
        return acc, loss_test


def train_citation(net, graph, features, labels, train_mask, test_mask, args):
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
        logits = net(graph, features)[0]
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # validation
        dur.append(time.time() - t0)
        acc, loss_test = evaluate_citation(net, graph, features, labels, test_mask)
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
    return acc


def evaluate_regression_citation(model, inputs, targets, mask, loss_fcn, args):
    # used to evaluate the performance of the model on test dataset
    model.eval()
    with torch.no_grad():
        predicted_outputs = model(inputs)
        if args.regression_metric == 'l2':
            loss_test = loss_fcn(predicted_outputs[mask], targets[mask])
        elif args.regression_metric == 'cos':
            y = torch.ones(targets[mask].shape[0])
            loss_test = loss_fcn(predicted_outputs[mask], targets[mask], y)
        return loss_test


def train_regression_citation(net, inputs, targets, train_mask, test_mask, args):
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
    if args.regression_metric == 'l2':
        loss_fcn = nn.MSELoss()
    elif args.regression_metric == 'cos':
        loss_fcn = nn.CosineEmbeddingLoss()
    dur = []

    for epoch in range(max_epoch):
        t0 = time.time()

        net.train()
        predicted_outputs = net(inputs)
        if args.regression_metric == 'l2':
            loss = loss_fcn(predicted_outputs[train_mask], targets[train_mask])
        elif args.regression_metric == 'cos':
            y = torch.ones(targets[train_mask].shape[0])
            loss = loss_fcn(predicted_outputs[train_mask], targets[train_mask], y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        dur.append(time.time() - t0)
        loss_test = evaluate_regression_citation(net, inputs, targets, test_mask, loss_fcn, args)
        print("Epoch {:04d} | Loss {} | Test Loss {} | Time(s) {:.4f}".format(epoch+1, loss.item(), loss_test, np.mean(dur)))

        # early stop
        if best_loss > loss_test:
            best_loss = np.min((best_loss, loss_test))
            cur_step = 0
        else:
            cur_step += 1
            if cur_step == patience:
                break


def evaluate_degree_regression_citation(model, inputs, targets, mask, loss_fcn):
    # used to evaluate the performance of the model on test dataset
    model.eval()
    with torch.no_grad():
        predicted_degrees = model(inputs)
        loss = loss_fcn(predicted_degrees[mask], targets[mask])
        upper_bound = 1.1 * targets[mask]
        lower_bound = 0.9 * targets[mask]
        acc = torch.sum((predicted_degrees[mask]<=upper_bound) & (predicted_degrees[mask]>=lower_bound)) * 1.0 / len(targets[mask])
        return acc, loss


def train_degree_regression_citation(net, inputs, targets, train_mask, test_mask, args):
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
    loss_fcn = nn.MSELoss()
    dur = []

    for epoch in range(max_epoch):
        t0 = time.time()

        net.train()
        predicted_outputs = net(inputs)
        loss = loss_fcn(predicted_outputs[train_mask], targets[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        dur.append(time.time() - t0)
        acc_test, loss_test = evaluate_degree_regression_citation(net, inputs, targets, test_mask, loss_fcn)
        print("Epoch {:04d} | Loss {} | Test Accuracy {} | Time(s) {:.4f}".format(epoch+1, loss.item(), acc_test, np.mean(dur)))

        # early stop
        if acc_test > best_accuracy or best_loss > loss_test:
            best_accuracy = np.max((acc_test, best_accuracy))
            best_loss = np.min((best_loss, loss_test))
            cur_step = 0
        else:
            cur_step += 1
            if cur_step == patience:
                break


def evaluate_ppi(model, valid_dataloader, loss_fcn):
    score_list = []
    val_loss_list = []
    for batch, (subgraph, labels) in enumerate(valid_dataloader):
        model.eval()
        with torch.no_grad():
            output = model(subgraph, subgraph.ndata['feat'].float().to(device))[0]
            loss_data = loss_fcn(output, labels.float().to(device)).item()
            predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
            score = f1_score(labels.data.cpu().numpy(), predict, average='micro')
        score_list.append(score)
        val_loss_list.append(loss_data)
    mean_score = np.array(score_list).mean()
    mean_val_loss = np.array(val_loss_list).mean()
    return mean_score, mean_val_loss


def train_ppi(net, train_dataloader, valid_dataloader):

    config_file = os.path.join(os.getcwd(), '../configs/ppi.yaml')
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    lr = config['train_lr']  # learning rate
    max_epoch = config['train_max_epoch']  # maximal number of training epochs
    # used for early stop
    patience = config['train_patience']
    best_score = -1
    best_loss = float('inf')
    cur_step = 0

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0)
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    dur = []

    for epoch in range(max_epoch):
        t0 = time.time()

        net.train()
        loss_list = []
        for batch, (subgraph, labels) in enumerate(train_dataloader):
            logits = net(subgraph, subgraph.ndata['feat'].float().to(device))[0]
            loss = loss_fcn(logits, labels.float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_data = np.array(loss_list).mean()

        # Validation
        dur.append(time.time() - t0)
        mean_score, mean_val_loss = evaluate_ppi(net, valid_dataloader, loss_fcn)
        print("Epoch {:04d} | Loss {:.4f} | Valid F1-Score {:.4f} | Time(s) {:.4f}".format(epoch + 1, loss_data, mean_score, np.mean(dur)))
        # early stop
        if mean_score > best_score or best_loss > mean_val_loss:
            best_score = np.max((mean_score, best_score))
            best_loss = np.min((best_loss, mean_val_loss))
            cur_step = 0
        else:
            cur_step += 1
            if cur_step == patience:
                break
    return mean_score


def evaluate_regression_ppi(model, valid_dataloader, loss_fcn, args):
    val_loss_list = []
    for batch, (subgraph, labels) in enumerate(valid_dataloader):
        model.eval()
        with torch.no_grad():
            predicted_output = model(subgraph.ndata['embedding'].float().to(device))
            if args.regression_metric == 'l2':
                loss_data = loss_fcn(predicted_output, subgraph.ndata['feat'].float())
            elif args.regression_metric == 'cos':
                y = torch.ones(subgraph.ndata['feat'].shape[0])
                loss_data = loss_fcn(predicted_output, subgraph.ndata['feat'].float(), y)
        val_loss_list.append(loss_data)
    mean_val_loss = np.array(val_loss_list).mean()
    return mean_val_loss


def train_regression_ppi(net, train_dataloader, valid_dataloader, args):

    config_file = os.path.join(os.getcwd(), '../configs/ppi.yaml')
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    lr = config['train_lr']  # learning rate
    max_epoch = config['train_max_epoch']  # maximal number of training epochs
    # used for early stop
    patience = config['train_patience']
    best_loss = float('inf')
    cur_step = 0

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0)
    if args.regression_metric == 'l2':
        loss_fcn = nn.MSELoss()
    elif args.regression_metric == 'cos':
        loss_fcn = nn.CosineEmbeddingLoss()
    dur = []

    for epoch in range(max_epoch):
        t0 = time.time()

        net.train()
        loss_list = []
        for batch, (subgraph, labels) in enumerate(train_dataloader):
            predicted_ouput = net(subgraph.ndata['embedding'].float().to(device))
            if args.regression_metric == 'l2':
                loss = loss_fcn(predicted_ouput, subgraph.ndata['feat'].float())
            elif args.regression_metric == 'cos':
                y = torch.ones(subgraph.ndata['feat'].shape[0])
                loss = loss_fcn(predicted_ouput, subgraph.ndata['feat'].float(), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

        mean_loss = np.array(loss_list).mean()

        # Validation
        dur.append(time.time() - t0)
        mean_val_loss = evaluate_regression_ppi(net, valid_dataloader, loss_fcn, args)
        print("Epoch {:04d} | Loss {} | Valid Loss {} | Time(s) {:.4f}".format(epoch + 1, mean_loss, mean_val_loss, np.mean(dur)))
        # early stop
        if best_loss > mean_val_loss:
            best_loss = np.min((best_loss, mean_val_loss))
            cur_step = 0
        else:
            cur_step += 1
            if cur_step == patience:
                break
