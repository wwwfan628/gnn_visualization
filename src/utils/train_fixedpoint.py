import torch
import time
import numpy as np
import torch.nn.functional as F
import yaml
import os
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(net, graph, features, labels, mask):
    # used to evaluate the performance of the model on test dataset
    net.eval()
    with torch.no_grad():
        logits = net(graph, features)[0]
        logp = F.log_softmax(logits, 1)
        loss_test = F.nll_loss(logp[mask], labels[mask])
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        acc = correct.item() * 1.0 / len(labels)
        return acc, loss_test


def train(net, graph, features, labels, train_mask, test_mask, args):
    writer = SummaryWriter(logdir='../logs/fixedpoint_' + args.dataset)

    # read parameters for training
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

    if args.fixpoint_loss:
        sigma_classification = torch.tensor([1.0])
        sigma_fixpoint = torch.tensor([1.0])
        parameters = list(net.parameters())+list(sigma_classification)+list(sigma_fixpoint)
        optimizer = torch.optim.Adam(parameters, lr=lr)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    dur = []   # duration for training epochs

    for epoch in range(max_epoch):
        t0 = time.time()  # start time

        net.train()

        if args.fixpoint_loss:
            net_outputs = net(graph, features)
            logits = net_outputs[0]
            logp = F.log_softmax(logits, 1)
            loss_1 = F.nll_loss(logp[train_mask], labels[train_mask])   # normal classification loss
            weighted_loss_1 = (1/(sigma_classification**2)) * loss_1   # weighted classification loss
            loss_2 = torch.mean(torch.abs(net_outputs[2] - net_outputs[1]))  # loss to find fixpoint
            weighted_loss_2 = (1/(2*sigma_fixpoint**2)) * loss_2   # weighted fixpoint loss
            # final loss function
            loss = weighted_loss_1 + weighted_loss_2 + torch.log(sigma_fixpoint * sigma_classification)
        else:
            logits = net(graph, features)[0]
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp[train_mask], labels[train_mask])  # normal classification loss

        # add loss values to tensorboard log
        writer.add_scalar('Training loss function value', loss, epoch)
        if args.fixpoint_loss:
            writer.add_scalar('Weighted classification loss', weighted_loss_1, epoch)
            writer.add_scalar('Actual classification loss', loss_1, epoch)
            writer.add_scalar('Weighted fixpoint loss', weighted_loss_2, epoch)
            writer.add_scalar('Actual fixpoint loss', loss_2, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        dur.append(time.time() - t0)
        acc, loss_test = evaluate(net, graph, features, labels, test_mask)
        if args.fixpoint_loss:
            print("Epoch {:04d} | Loss {:.4f} | Test Acc {:.4f} | Fixpoint Loss: {:.4f} | Weighted Fixpoint Loss: {:.4f} | Time(s) {:.4f}".format(
                epoch + 1, loss.item(), acc, loss_2.item(), weighted_loss_2.item(), np.mean(dur)))
        else:
            print("Epoch {:04d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(epoch + 1, loss.item(), acc, np.mean(dur)))

        # add test accuracy to tensorboard log
        writer.add_scalar('Test accuracy', acc, epoch)

        # early stop
        if acc > best_acc or best_loss > loss_test:
            best_acc = np.max((acc, best_acc))
            best_loss = np.min((best_loss, loss_test))
            cur_step = 0
        else:
            cur_step += 1
            if cur_step == patience:
                break

    return best_acc
