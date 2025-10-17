from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
import os
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.image as mpimg


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(400, 169),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(169, 49),
            nn.ReLU()
        )
        self.layer4 = nn.Linear(49, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class MyDataSet(Dataset):
    def __init__(self, datas: torch.tensor):
        self.datas = datas
    def __len__(self):
        return len(self.datas)
    def __getitem__(self, item):
        data = self.datas[item]
        return data

def hook(module, input, output):
    features.append(output.clone().detach())
    return None

dir_path = 'E:/course-code/meachine-learning/dataset/MNIST/train'
file_ls = os.listdir(dir_path)
xTraining = torch.zeros((60000,784))
yTraining = torch.zeros((60000, 10))
flag = 0
for dir in file_ls:
    files = os.listdir(dir_path+'\\'+dir)
    for file in files:
        filename = dir_path+'\\'+dir+'\\'+file
        img = mpimg.imread(filename)
        xTraining[flag,:] = torch.from_numpy(np.reshape(img, -1)/255)
        flag+=1

scaler = StandardScaler(copy=False)
scaler.fit(xTraining)
scaler.transform(xTraining)
net = NN()

class NN1(nn.Module):
    def __init__(self):
        super(NN1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU()
        )
        self.layer2 =  nn.Linear(400, 784)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
net1 = NN1()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net1.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
num_epochs = 100
batch_size = 100
train_loader = torch.utils.data.DataLoader(MyDataSet(xTraining),batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    totalLoss = 0
    for batch_x in train_loader:
        outputs = net1(batch_x)
        loss = criterion(outputs, batch_x)
        optimizer.zero_grad()
        loss.backward()
        totalLoss += loss.item()
        optimizer.step()
    print(totalLoss)

net.layer1 = net1.layer1
features = []
handle = net1.layer1.register_forward_hook(hook)
y = net1(xTraining)
xTraining = features[0]
print(xTraining.shape)

class NN2(nn.Module):
    def __init__(self):
        super(NN2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(400, 169),
            nn.ReLU()
        )
        self.layer2 =  nn.Linear(169, 400)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

net2 = NN2()
optimizer = torch.optim.Adam(net2.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
train_loader = torch.utils.data.DataLoader(MyDataSet(xTraining),batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    totalLoss = 0
    for batch_x in train_loader:
        outputs = net2(batch_x)
        loss = criterion(outputs, batch_x)
        optimizer.zero_grad()
        loss.backward()
        totalLoss += loss.item()
        optimizer.step()
    print(totalLoss)
net.layer2 = net2.layer1

features = []
handle = net2.layer1.register_forward_hook(hook)
y = net2(xTraining)
xTraining = features[0]
print(xTraining.shape)

class NN3(nn.Module):
    def __init__(self):
        super(NN3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(169, 49),
            nn.ReLU()
        )
        self.layer2 =  nn.Linear(49, 169)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

net3 = NN3()
optimizer = torch.optim.Adam(net3.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
train_loader = torch.utils.data.DataLoader(MyDataSet(xTraining),batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    totalLoss = 0
    for batch_x in train_loader:
        outputs = net3(batch_x)
        loss = criterion(outputs, batch_x)
        optimizer.zero_grad()
        loss.backward()
        totalLoss += loss.item()
        optimizer.step()
    print(totalLoss)
net.layer3 = net3.layer1

features = []
handle = net3.layer1.register_forward_hook(hook)
y = net3(xTraining)
xTraining = features[0]
print(xTraining.shape)

class NN4(nn.Module):
    def __init__(self):
        super(NN4, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(49, 10),
            nn.ReLU()
        )
        self.layer2 =  nn.Linear(10, 49)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

net4 = NN4()
optimizer = torch.optim.Adam(net4.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
train_loader = torch.utils.data.DataLoader(MyDataSet(xTraining),batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    totalLoss = 0
    for batch_x in train_loader:
        outputs = net4(batch_x)
        loss = criterion(outputs, batch_x)
        optimizer.zero_grad()
        loss.backward()
        totalLoss += loss.item()
        optimizer.step()
    print(totalLoss)
net.layer4.weight = net4.layer1[0].weight
net.layer4.bias = net4.layer1[0].bias
torch.save(net.state_dict(), 'modelAutoEncoder.pkl')