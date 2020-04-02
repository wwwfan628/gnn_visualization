import torch
import time
import numpy as np
import torch.nn.functional as F


def evaluate_train(model, graph, features, labels, mask):
    # used to evaluate the performance of the model on test dataset
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train_net(net, graph, features, labels, train_mask, test_mask, args):

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr_train)
    dur = []

    for epoch in range(args.epoch_train):
        t0 = time.time()

        net.train()
        logits = net(graph, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            dur.append(time.time() - t0)

            acc = evaluate_train(net, graph, features, labels, test_mask)
            print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
                    epoch, loss.item(), acc, np.mean(dur)))



