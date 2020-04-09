import torch
import time
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import yaml
import os
from sklearn.metrics import f1_score

def evaluate_cora_reddit(model, graph, features, labels, mask):
    # used to evaluate the performance of the model on test dataset
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logp = F.log_softmax(logits, 1)
        loss_test = F.nll_loss(logp[mask], labels[mask])
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        acc = correct.item() * 1.0 / len(labels)
        return acc, loss_test


def train_cora_reddit(net, graph, features, labels, train_mask, test_mask, args):
    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f)

    lr = config['train_lr']  # learning rate
    max_epoch = config['train_max_epoch']  # maximal number of training epochs
    # used for early stop
    patience = config['train_patience']
    best_score = -1
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

        if epoch % 5 == 0:  # Validation
            dur.append(time.time() - t0)
            acc, loss_valid = evaluate_cora_reddit(net, graph, features, labels, test_mask)
            print("Epoch {:04d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(epoch+1, loss.item(), acc, np.mean(dur)))

            # early stop
            if acc > best_score or best_loss > loss_valid:
                best_score = np.max((acc, best_score))
                best_loss = np.min((best_loss, loss_valid))
                cur_step = 0
            else:
                cur_step += 1
                if cur_step == patience:
                    break



def evaluate_ppi(model, valid_dataloader, loss_fcn):
    score_list = []
    val_loss_list = []
    for batch, (subgraph, feats, labels) in enumerate(valid_dataloader):
        model.eval()
        with torch.no_grad():
            output = model(subgraph, feats.float())
            loss_data = loss_fcn(output, labels.float()).item()
            predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
            score = f1_score(labels.data.cpu().numpy(), predict, average='micro')
        score_list.append(score)
        val_loss_list.append(loss_data)
    mean_score = np.array(score_list).mean()
    mean_val_loss = np.array(val_loss_list).mean()
    return mean_score, mean_val_loss


def train_ppi(net, train_dataloader, valid_dataloader, args):

    config_file = os.path.join(os.getcwd(), '../configs/ppi.yaml')
    with open(config_file, 'r') as f:
        config = yaml.load(f)

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
        for batch, (subgraph, feats, labels) in enumerate(train_dataloader):
            logits = net(subgraph, feats.float())
            loss = loss_fcn(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_data = np.array(loss_list).mean()

        if epoch % 5 == 0:  # Validation
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



def evaluate_tu(valid_dataloader, model, loss_fcn, batch_size):
    val_loss_list = []
    correct_label = 0
    for batch_idx, (batch_graph, graph_labels) in enumerate(valid_dataloader):
        model.eval()
        with torch.no_grad():
            ypred = model(batch_graph, batch_graph.ndata['feat'])
            loss = loss_fcn(ypred, graph_labels).item()
            indi = torch.argmax(ypred, dim=1)
            correct = torch.sum(indi == graph_labels)
            correct_label += correct.item()
        val_loss_list.append(loss)
    mean_val_loss = np.array(val_loss_list).mean()
    acc = correct_label / (len(valid_dataloader) * batch_size)
    return acc, mean_val_loss


def train_tu(net, train_dataloader, valid_dataloader, args):

    config_file = os.path.join(os.getcwd(), '../configs/tu.yaml')
    with open(config_file, 'r') as f:
        config = yaml.load(f)

    lr = config['train_lr']  # learning rate
    max_epoch = config['train_max_epoch']  # maximal number of training epochs
    clip = config['clip']
    batch_size = config['batch_size']
    # used for early stop
    patience = config['train_patience']
    best_score = -1
    best_loss = float('inf')
    cur_step = 0

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fcn = nn.CrossEntropyLoss()
    dur = []

    for epoch in range(max_epoch):
        t0 = time.time()

        net.train()
        loss_list = []
        for (batch_idx, (batch_graph, graph_labels)) in enumerate(train_dataloader):
            net.zero_grad()
            ypred = net(batch_graph, batch_graph.ndata['feat'])
            loss = loss_fcn(ypred, graph_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()
            loss_list.append(loss.item())
        loss_data = np.array(loss_list).mean()

        if epoch % 5 == 0:  # validation
            dur.append(time.time() - t0)
            acc, mean_val_loss = evaluate_tu(valid_dataloader, net, loss_fcn, batch_size)
            print("Epoch {:04d} | Loss {:.4f} | Valid Acc {:.4f} | Time(s) {:.4f}".format(epoch + 1, loss_data, acc, np.mean(dur)))
            # early stop
            if acc > best_score or best_loss > mean_val_loss:
                best_score = np.max((acc, best_score))
                best_loss = np.min((best_loss, mean_val_loss))
                cur_step = 0
            else:
                cur_step += 1
                if cur_step == patience:
                    break

def load_parameters(file, net):

    pretrained_dict = torch.load(file)
    model_dict = net.state_dict()

    # filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    return model_dict
