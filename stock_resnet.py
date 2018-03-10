import argparse
import math
import os
import shutil

import numpy as np
import torch
from progressbar import ProgressBar
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils import data as data_utils

N_ITER = 0

class StockDataLoader(data_utils.TensorDataset):
    def __init__(self, dataX, dataY):
        self.dataX = dataX
        self.dataY = dataY
        self.length = dataX.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.dataX[idx, ...]),\
               torch.from_numpy(self.dataY[np.newaxis, idx, ...])

    def __len__(self):
        return self.length

def load_dataset(x_path, y_path, batch_size=300, num_workers=2):
    print("loading dataset")
    stock_data = StockDataLoader(np.load(x_path),
                                 np.load(y_path))
    print("Making dataset loader")
    stock_loader = data_utils.DataLoader(stock_data,
                                         batch_size = batch_size,
                                         shuffle = True,
                                         num_workers = num_workers)
    return stock_loader

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def Resnet50():
    model = ResNet(Bottleneck, [3,4,6,3], num_classes=1)
    return model

def loss_func(output, target):
    loss = F.mse_loss(output, target)
    return loss

def train(model, train_loader, optimizer, epoch, writer, loss_tr, log_interval=100):
    global N_ITER
    model.train()
    bar = ProgressBar()
    for batch_idx, (data, target) in bar(enumerate(train_loader)):
        N_ITER += 1
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            writer.add_scalar(loss_tr, loss.data[0], N_ITER)

def validate(model, test_loader, epoch, writer, loss_va):
    model.eval()
    test_loss = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += loss_func(output, target).data[0]

    avg_loss = test_loss/len(test_loader.dataset)
    writer.add_scalar(loss_va, avg_loss, epoch)
    return avg_loss

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Stock data trainer')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for input trianing data, default=64')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to be train, defualt=10')
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--name', type=str, metavar='Name',
                        help='name to save in log directory')
    args = parser.parse_args()

    CUR_DIR = os.getcwd()
    log_dir = os.path.join(CUR_DIR, 'log', args.name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    x_train_Path = os.path.join(CUR_DIR,'data','trX.npy')
    y_train_path = os.path.join(CUR_DIR,'data','trY.npy')
    train_loader = load_dataset(x_train_Path, y_train_path,
                                batch_size=args.batch_size)

    x_val_Path = os.path.join(CUR_DIR,'data','teX.npy')
    y_val_path = os.path.join(CUR_DIR,'data','teY.npy')
    val_loader = load_dataset(x_val_Path, y_val_path,
                                batch_size=args.batch_size)

    model = Resnet50().cuda()

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr)

    writer = SummaryWriter()
    loss_tr = os.path.join(log_dir,'data','l_tr')
    loss_va = os.path.join(log_dir,'data','l_va')
    min_loss = float('inf')
    for ep in range(args.epochs):
        print("Starting epoch %d"%ep)
        train(model, train_loader, optimizer, args.batch_size, writer, loss_tr)
        ep_loss = validate(model, val_loader, ep, writer, loss_va)

        is_best = False
        if ep_loss < min_loss:
            is_best = True
            min_loss = ep_loss
        save_checkpoint({
                'epoch': ep,
                'state_dict': model.state_dict(),
                'best_acc': ep_loss,
                'optimizer' : optimizer.state_dict(),
                }, is_best,
                filename="{}/checkpoint.{}th.tar".format(log_dir, ep))
