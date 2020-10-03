import torch as T
import torch.nn as nn
import torch.nn.functional as F

class CnnBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel):
        super(CnnBlock,self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride=2)
        # self.drop = nn.Dropout(0.2)
        self.pool = nn.MaxPool2d(3, 2)
        self.bn = nn.BatchNorm2d(out_channel)

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        # # x = self.bn(self.drop(x))
        x = self.bn(x)

        #batch norm before relu
        # x = self.bn(self.conv(x))
        # x = F.relu(x)
        # x = self.pool(x)
        return x


class XrayNet(nn.Module):
    def __init__(self):
        super(XrayNet,self).__init__()
        self.cnn1 = CnnBlock(1,16,3)
        self.cnn2 = CnnBlock(16,32,3)
        self.cnn3 = CnnBlock(32, 64, 3)
        # self.cnn4 = CnnBlock(64, 128, 3)

        # print(self.cnn3.size)
        #1536 3 2
        self.fc1 = nn.Linear(1536, 1024)
        self.fc2 = nn.Linear(1024, 2)

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        # print(x.shape)
        x = self.cnn3(x)
        # print(x.shape)
        #flatten
        x = x.reshape(x.shape[0], -1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.softmax(x,1)

