import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class net_mnist(nn.Module):
    def __init__(self):
        super(net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=8, padding=3, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6, padding=2, stride=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(1152, 10)

    def forward(self, x):
        out1 = self.relu1(self.conv1(x))
        out2 = self.relu2(self.conv2(out1))
        out3 = self.relu3(self.conv3(out2))
        out3_ = out3.view(out3.size(0), -1) 
        out4 = self.linear1(out3_)

        return out1, out2, out3, out4

class knn_cnn(nn.Module):
    def __init__(self):
        super(knn_cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=8, padding=3, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6, padding=2, stride=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(1152, 10)

        self.knn_1 = nn.Linear(12544, 10)
        self.knn_2 = nn.Linear(6272, 10)
        self.knn_3 = nn.Linear(1152, 10)
        self.knn_4 = nn.Linear(10, 10)


    def forward(self, x):
        out1 = self.relu1(self.conv1(x))
        out2 = self.relu2(self.conv2(out1))
        out3 = self.relu3(self.conv3(out2))
        out3_ = out3.view(out3.size(0), -1) 
        out4 = self.linear1(out3_)

        knn1 = self.knn_1(out1.view(out1.size(0), -1))
        knn2 = self.knn_2(out2.view(out2.size(0), -1))
        knn3 = self.knn_3(out3.view(out3.size(0), -1))
        knn4 = self.knn_4(out4)


        return out1, out2, out3, out4, knn1, knn2, knn3, knn4

'''
class MNIST_dataset(torchvision.datasets.MNIST):
    def __init__(self, targets):
        self.targets_ = targets
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets_[index])
'''

class net_svhn(nn.Module):
    def __init__(self):
        super(net_svhn, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=8, padding=3, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6, padding=2, stride=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(2048, 10)

    def forward(self, x):
        out1 = self.relu1(self.conv1(x))
        out2 = self.relu2(self.conv2(out1))
        out3 = self.relu3(self.conv3(out2))
        out3_ = out3.view(out3.size(0), -1) 
        out4 = self.linear1(out3_)

        return out1, out2, out3, out4

class knn_cnn_svhn(nn.Module):
    def __init__(self):
        super(knn_cnn_svhn, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=8, padding=3, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6, padding=2, stride=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(2048, 10)

        self.knn_1 = nn.Linear(16384, 10)
        self.knn_2 = nn.Linear(8192, 10)
        self.knn_3 = nn.Linear(2048, 10)
        self.knn_4 = nn.Linear(10, 10)


    def forward(self, x):
        out1 = self.relu1(self.conv1(x))
        out2 = self.relu2(self.conv2(out1))
        out3 = self.relu3(self.conv3(out2))
        out3_ = out3.view(out3.size(0), -1) 
        out4 = self.linear1(out3_)

        knn1 = self.knn_1(out1.view(out1.size(0), -1))
        knn2 = self.knn_2(out2.view(out2.size(0), -1))
        knn3 = self.knn_3(out3.view(out3.size(0), -1))
        knn4 = self.knn_4(out4)


        return out1, out2, out3, out4, knn1, knn2, knn3, knn4

