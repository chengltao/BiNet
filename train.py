import torch.nn.functional as F
import torch.optim as optim
import os
import torch
import datetime
import numpy as np
from torch.utils.data import DataLoader
from load_data import MyDatasets
from test import test

def train(args, model, device, train_file, test_file):
    #loading training data
    train_loader = DataLoader(MyDatasets(train_file), batch_size=args['batch_size'], shuffle=True)
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
    # iterating
    for epoch in range(1, args['epochs'] + 1):


        train_epoch(epoch, model, device, train_loader, optimizer)
        test(args, model, device, test_file)

    return model


#trainging one epoch
def train_epoch(epoch, model, device, data_loader, optimizer):
    # switch to trianing mode
    model.train()
    pid = os.getpid()
    sum_loss = 0.0
    count = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        count = count + 1
        optimizer.zero_grad()
        output = model(data.float().to(device))
        # computing loss
        loss = F.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()
        sum_loss = sum_loss + loss.item()
        if batch_idx % 10 == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                            100. * batch_idx / len(data_loader), loss.item()))

    return sum_loss/count



