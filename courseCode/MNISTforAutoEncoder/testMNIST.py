from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
import os
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.image as mpimg

def compute_accuracy(net,X,Y,batch_size=100):
    data_loader = torch.utils.data.DataLoader(MyDataSet(X,Y), batch_size=batch_size, shuffle=False)
    accuracy = 0
    for batch_x,batch_y in data_loader:
        outputs = net(batch_x)
        accuracy += (outputs.argmax(dim=1) == batch_y.argmax(dim=1)).float().sum().item()
    return accuracy/len(X)

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

net = NN()
if os.path.exists('modelAutoEncoder.pkl'):
    net.load_state_dict(torch.load('modelAutoEncoder.pkl', map_location='cpu'))

class MyDataSet(Dataset):
    def __init__(self, datas,labels: torch.tensor):
        self.datas = datas
        self.labels = labels
    def __len__(self):
        return len(self.datas)
    def __getitem__(self, item):
        data = self.datas[item]
        label = self.labels[item]
        return data,label

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
        yTraining[flag, int(dir)] = 1
        flag+=1

dir_path = 'E:/course-code/meachine-learning/dataset/MNIST/test'
file_ls = os.listdir(dir_path)
xTesting = torch.zeros((10000,784))
yTesting = torch.zeros((10000, 10))
flag = 0
for dir in file_ls:
    files = os.listdir(dir_path+'\\'+dir)
    for file in files:
        filename = dir_path+'\\'+dir+'\\'+file
        img = mpimg.imread(filename)
        xTesting[flag,:] = torch.from_numpy(np.reshape(img, -1)/255)
        yTesting[flag, int(dir)] = 1
        flag+=1
ratioTraining = 0.95
ratioValidation = 0.01
xTraining, xValidation, yTraining, yValidation = train_test_split(xTraining, yTraining, test_size=1 - ratioTraining, random_state=1)  # 随机分配数据集

scaler = StandardScaler(copy=False)
scaler.fit(xTraining)
scaler.transform(xTraining)
scaler.transform(xValidation)
scaler.transform(xTesting)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
num_epochs = 50
batch_size = 100
train_loader = torch.utils.data.DataLoader(MyDataSet(xTraining,yTraining),batch_size=batch_size, shuffle=True)
maxAccuracy = 0
for epoch in range(num_epochs):
    avgLoss = 0
    for batch_x,batch_y in train_loader:
        outputs = net(batch_x)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        avgLoss += float(loss.item())
        optimizer.step()
    accuracy = compute_accuracy(net, xValidation, yValidation, batch_size)
    print('epoch:%d, accuracy:%.3f' % (epoch + 1, accuracy))
    if accuracy > maxAccuracy:
        maxAccuracy = accuracy
        torch.save(net.state_dict(), 'model.pkl')

net = NN()
if os.path.exists('model.pkl'):
    net.load_state_dict(torch.load('model.pkl', map_location='cpu'))
    batch_size = 100
    accuracy = compute_accuracy(net, xTesting, yTesting, batch_size)
    print('The accuracy on the testing set is %.3f%%' % (accuracy*100))

