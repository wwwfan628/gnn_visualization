import torch
import time
import numpy as np
import torch.nn.functional as F
import yaml
import os
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    writer = SummaryWriter(logdir='../logs/' + args.dataset + '_without_loss2')

    path = '../configs/' + args.dataset + '.yaml'
    config_file = os.path.join(os.getcwd(), path)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    lr = config['train_lr']  # learning rate
    max_epoch = config['train_max_epoch']  # maximal number of training epochs
    t_update = 5
    t_optimize = 1
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
        loss = F.nll_loss(logp[train_mask], labels[train_mask])   # normal training loss

        writer.add_scalar('Training loss function value', loss, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0:  # Validation
            dur.append(time.time() - t0)
            acc, loss_valid = evaluate_cora_reddit(net, graph, features, labels, test_mask)
            print("Epoch {:04d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(epoch+1, loss.item(), acc, np.mean(dur)))
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


def train_ppi(net, train_dataloader, valid_dataloader, args):
    writer = SummaryWriter(logdir='../logs/' + args.dataset + '_without_loss2')

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
            net_outputs = net(subgraph, subgraph.ndata['feat'].float().to(device))
            logits = net_outputs[0]
            loss_1 = loss_fcn(logits, labels.float().to(device))
            # loss to find fixpoint
            loss_2 = torch.mean(torch.abs(net_outputs[2] - net_outputs[1]))
            # final loss function
            if args.loss_weight:
                loss = (1 / (sigma_classification ** 2)) * loss_1 + (1 / (2 * sigma_fixpoint ** 2)) * loss_2 + torch.log(sigma_fixpoint * sigma_classification)
            elif args.without_fixpoint_loss:
                loss = loss_1
            else:
                loss = loss_1 + loss_2

            writer.add_scalar('Training loss function value, loss_weight=' + str(args.loss_weight), loss, epoch)
            if args.loss_weight:
                writer.add_scalar('Training fixpoint loss, loss_weight=' + str(args.loss_weight),
                                  (1 / (2 * sigma_fixpoint ** 2)) * loss_2, epoch)
                writer.add_scalar('Actual fixpoint loss, loss_weight=' + str(args.loss_weight), loss_2, epoch)
            else:
                writer.add_scalar('Training fixpoint loss, loss_weight=' + str(args.loss_weight), loss_2, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_data = np.array(loss_list).mean()

        if epoch % 1 == 0:  # Validation
            dur.append(time.time() - t0)
            mean_score, mean_val_loss = evaluate_ppi(net, valid_dataloader, loss_fcn)
            print("Epoch {:04d} | Loss {:.4f} | Valid F1-Score {:.4f} | Time(s) {:.4f}".format(epoch + 1, loss_data, mean_score, np.mean(dur)))
            writer.add_scalar('Micro-F1 Score, loss_weight=' + str(args.loss_weight), mean_score, epoch)
            # early stop
            if mean_score > best_score or best_loss > mean_val_loss:
                best_score = np.max((mean_score, best_score))
                best_loss = np.min((best_loss, mean_val_loss))
                cur_step = 0
            else:
                cur_step += 1
                if cur_step == patience:
                    break