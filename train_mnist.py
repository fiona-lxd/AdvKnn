import matplotlib
matplotlib.use('agg')

import pdb

import matplotlib.pyplot as plt
import os
import argparse
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader, get_fashion_mnist_test_loader
from advertorch_examples.utils import _imshow
import pickle

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

from net import net_mnist
from advertorch_examples.utils import TRAINED_MODEL_PATH
TRAINED_MODEL_PATH = "./fashion_mnist/"
model_filename  = "fashion_mnist_lenet5_clntrained.pt"


model = net_mnist()
model.to(device)

net = 'fashion_mnist'

from advertorch_examples.utils import get_mnist_train_loader, get_fashion_mnist_train_loader
if net == 'fashion_mnist':
    train_loader = get_fashion_mnist_train_loader(
            batch_size=128, shuffle=True)
    test_loader = get_fashion_mnist_test_loader(batch_size=1000, shuffle=False)

else:
    train_loader = get_mnist_train_loader(
            batch_size=128, shuffle=True)
    test_loader = get_mnist_test_loader(batch_size=1000, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

nb_epoch = 18
log_interval = 50

tmp_correct = 0
for epoch in range(nb_epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # pdb.set_trace()
        output = model(data)
        # loss = F.cross_entropy(output[-1], target, reduction='elementwise_mean')
        loss = F.cross_entropy(output[-1], target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx *
                    len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    model.eval()
    test_clnloss = 0
    clncorrect = 0

    for clndata, target in test_loader:
        clndata, target = clndata.to(device), target.to(device)
        with torch.no_grad():
            output = model(clndata)
        # pdb.set_trace()
        test_clnloss += F.cross_entropy(output[-1], target, reduction='sum').item()
        pred = output[-1].max(1, keepdim=True)[1]
        clncorrect += pred.eq(target.view_as(pred)).sum().item()

    test_clnloss /= len(test_loader.dataset)
    print('\nTest set: avg cln loss: {:.4f},'
              ' cln acc: {}/{} ({:.0f}%)\n'.format(
                  test_clnloss, clncorrect, len(test_loader.dataset),
                  100. * clncorrect / len(test_loader.dataset)))
    if clncorrect > tmp_correct:
        tmp_correct = clncorrect
        print(tmp_correct)

        torch.save(
            model.state_dict(),
            os.path.join(TRAINED_MODEL_PATH, model_filename))








