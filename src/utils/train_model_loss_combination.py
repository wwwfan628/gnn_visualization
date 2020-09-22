import torch
import time
import numpy as np
import torch.nn.functional as F
import yaml
import os
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_citation(model, graph, features, labels, mask, args):
    # used to evaluate the performance of the model on test dataset
    model.eval()
    with torch.no_grad():
        if args.loss_weight:
            logits = model(graph, features)[0]
        elif args.without_fixpoint_loss:
            logits = model(graph, features)
        logp = F.log_softmax(logits, 1)
        loss_test = F.nll_loss(logp[mask], labels[mask])
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        acc = correct.item() * 1.0 / len(labels)
        return acc, loss_test


def train_citation(net, graph, features, labels, train_mask, test_mask, args):
    writer = SummaryWriter(logdir='../logs/' + args.dataset + 'loss_weight=' + str(args.loss_weight))

    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    lr = config['train_lr']  # learning rate
    max_epoch = config['train_max_epoch']  # maximal number of training epochs
    # used for early stop
    patience = config['train_patience']
    best_score = -1
    best_loss = float('inf')
    cur_step = 0

    if args.loss_weight:
        sigma_classification = torch.tensor([1.0])
        sigma_fixpoint = torch.tensor([1.0])
        parameters = list(net.parameters())+list(sigma_classification)+list(sigma_fixpoint)
        optimizer = torch.optim.Adam(parameters, lr=lr)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    dur = []

    for epoch in range(max_epoch):
        t0 = time.time()

        net.train()

        if args.loss_weight:
            net_outputs = net(graph, features)
            logits = net_outputs[0]
            logp = F.log_softmax(logits, 1)
            loss_1 = F.nll_loss(logp[train_mask], labels[train_mask])   # normal training loss
            weighted_loss_1 = (1/(sigma_classification**2)) * loss_1
            # loss to find fixpoint
            loss_2 = torch.mean(torch.abs(net_outputs[2] - net_outputs[1]))
            weighted_loss_2 = (1/(2*sigma_fixpoint**2)) * loss_2
            # final loss function
            loss = weighted_loss_1 + weighted_loss_2 + torch.log(sigma_fixpoint * sigma_classification)
        elif args.without_fixpoint_loss:
            logits = net(graph, features)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp[train_mask], labels[train_mask])  # normal training loss

        writer.add_scalar('Training loss function value', loss, epoch)
        if args.loss_weight:
            writer.add_scalar('Weighted classification loss', weighted_loss_1, epoch)
            writer.add_scalar('Actual classification loss', loss_1, epoch)
            writer.add_scalar('Weighted fixpoint loss', weighted_loss_2, epoch)
            writer.add_scalar('Actual fixpoint loss', loss_2, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0:  # Validation
            dur.append(time.time() - t0)
            acc, loss_valid = evaluate_citation(net, graph, features, labels, test_mask, args)
            if args.loss_weight:
                print(
                    "Epoch {:04d} | Loss {:.4f} | Test Acc {:.4f} | Fixpoint Loss: {:.4f} | Weighted Fixpoint Loss: {:.4f} | Time(s) {:.4f}".format(
                        epoch + 1, loss.item(), acc, loss_2.item(), weighted_loss_2.item(), np.mean(dur)))
            elif args.without_fixpoint_loss:
                print("Epoch {:04d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(epoch + 1, loss.item(), acc, np.mean(dur)))

            writer.add_scalar('Test accuracy', acc, epoch)

            # early stop
            if acc > best_score or best_loss > loss_valid:
                best_score = np.max((acc, best_score))
                best_loss = np.min((best_loss, loss_valid))
                cur_step = 0
            else:
                cur_step += 1
                if cur_step == patience:
                    break



def evaluate_ppi(model, valid_dataloader, loss_fcn, args):
    score_list = []
    val_loss_list = []
    for batch, (subgraph, labels) in enumerate(valid_dataloader):
        model.eval()
        with torch.no_grad():
            if args.loss_weight:
                output = model(subgraph, subgraph.ndata['feat'].float().to(device))[0]
            elif args.without_fixpoint_loss:
                output = model(subgraph, subgraph.ndata['feat'].float().to(device))
            loss_data = loss_fcn(output, labels.float().to(device)).item()
            predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
            score = f1_score(labels.data.cpu().numpy(), predict, average='micro')
        score_list.append(score)
        val_loss_list.append(loss_data)
    mean_score = np.array(score_list).mean()
    mean_val_loss = np.array(val_loss_list).mean()
    return mean_score, mean_val_loss


def train_ppi(net, train_dataloader, valid_dataloader, args):
    writer = SummaryWriter(logdir='../logs/' + args.dataset + 'loss_weight=' + str(args.loss_weight))

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

    if args.loss_weight:
        sigma_classification = torch.tensor([1.0])
        sigma_fixpoint = torch.tensor([1.0])
        parameters = list(net.parameters())+list(sigma_classification)+list(sigma_fixpoint)
        optimizer = torch.optim.Adam(parameters, lr=lr)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0)
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    dur = []

    for epoch in range(max_epoch):
        t0 = time.time()

        net.train()
        loss_list = []
        for batch, (subgraph, labels) in enumerate(train_dataloader):
            if args.loss_weight:
                net_outputs = net(subgraph, subgraph.ndata['feat'].float().to(device))
                logits = net_outputs[0]
                loss_1 = loss_fcn(logits, labels.float().to(device))
                weighted_loss_1 = (1 / (sigma_classification ** 2)) * loss_1
                # loss to find fixpoint
                loss_2 = torch.mean(torch.abs(net_outputs[2] - net_outputs[1]))
                weighted_loss_2 = (1 / (2 * sigma_fixpoint ** 2)) * loss_2
                # final loss function
                loss = weighted_loss_1 + weighted_loss_2 + torch.log(sigma_fixpoint * sigma_classification)
            elif args.without_fixpoint_loss:
                logits = net(subgraph, subgraph.ndata['feat'].float().to(device))
                loss = loss_fcn(logits, labels.float().to(device))

            writer.add_scalar('Training loss function value', loss, epoch)
            if args.loss_weight:
                writer.add_scalar('Weighted classification loss', weighted_loss_1, epoch)
                writer.add_scalar('Actual classification loss', loss_1, epoch)
                writer.add_scalar('Training fixpoint loss', weighted_loss_2, epoch)
                writer.add_scalar('Actual fixpoint loss', loss_2, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        if epoch % 1 == 0:  # Validation
            dur.append(time.time() - t0)
            mean_score, mean_val_loss = evaluate_ppi(net, valid_dataloader, loss_fcn, args)
            if args.loss_weight:
                print("Epoch {:04d} | Loss {:.4f} | F1-Score {:.4f} | Fixpoint Loss {:.4f} | Weighted Fixpoint Loss {:.4f} | Time(s) {:.4f}".format(
                        epoch + 1, mean_val_loss, mean_score, loss_2, weighted_loss_2.item(), np.mean(dur)))
            else:
                print("Epoch {:04d} | Loss {:.4f} | F1-Score {:.4f} | Time(s) {:.4f}".format(epoch + 1, mean_val_loss, mean_score, np.mean(dur)))

            writer.add_scalar('Micro-F1 Score', mean_score, epoch)
            # early stop
            if mean_score > best_score or best_loss > mean_val_loss:
                best_score = np.max((mean_score, best_score))
                best_loss = np.min((best_loss, mean_val_loss))
                cur_step = 0
            else:
                cur_step += 1
                if cur_step == patience:
                    break


def load_parameters(file, net):

    pretrained_dict = torch.load(file, map_location=device)
    model_dict = net.state_dict()

    # filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    return model_dict
