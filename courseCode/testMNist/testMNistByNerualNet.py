import torch
from torch import nn, optim
from torch.nn import init
from torchvision import datasets, transforms
from torch.utils import data
import torch.nn.functional as F

import sys

use_gpu = torch.cuda.is_available()

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Sigmoid(), #也可 nn.Sigmoid()
            nn.AvgPool2d(2,2), #也可nn.AvgPool2d(2,2)
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(), #也可 nn.Sigmoid()
            nn.AvgPool2d(2,2) #也可nn.AvgPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),#也可 nn.Sigmoid()
            nn.Linear(120, 84),
            nn.Sigmoid(),#也可 nn.Sigmoid()
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = feature.view(img.shape[0], -1)
        output = self.fc(output)
        return output

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), #in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3,2), #kernel_size, stride
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),#也可 nn.Sigmoid()
            nn.Linear(4096, 4096),
            nn.ReLU(),#也可 nn.Sigmoid()
            nn.Linear(4096, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class ResNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        return F.relu(extra_x + output)


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            ResNetBasicBlock(64, 64, 1),
            ResNetBasicBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(
            ResNetDownBlock(64, 128, [2, 1]),
            ResNetBasicBlock(128, 128, 1)
        )
        self.layer3 = nn.Sequential(
            ResNetDownBlock(128, 256, [2, 1]),
            ResNetBasicBlock(256, 256, 1)
        )
        self.layer4 = nn.Sequential(
            ResNetDownBlock(256, 512, [2, 1]),
            ResNetBasicBlock(512, 512, 1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 10)

    def forward(self, img):
        output = self.conv1(img)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avgpool(output)
        output = output.reshape(img.shape[0], -1)
        output = self.fc(output)
        return output

if __name__=='__main__':
    net = LeNet()
    if (use_gpu):
        net.cuda()

    validation_ratio = 0.1
    batch_size = 100
    dataset = datasets.ImageFolder(root = 'E:/course-code/meachine-learning/dataset/MNIST/train', transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor()]))
    validation_number = round(validation_ratio*len(dataset))
    validationdataset,trainingdataset = data.random_split(dataset,lengths=[validation_number,len(dataset)-validation_number], generator=torch.Generator().manual_seed(0))
    train_iter = data.DataLoader(trainingdataset, batch_size = batch_size, shuffle = True )
    validation_iter = data.DataLoader(validationdataset, batch_size = batch_size, shuffle = False )

    if (use_gpu):
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    #optimizer = torch.optim.SGD(net.parameters(),lr=1e-3)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9,0.999), eps=1e-08, weight_decay=0)
    num_epochs = 50

    maxRecognitionRate = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        number = 0
        for inputs, labels in train_iter:
            if (use_gpu):
                inputs = inputs.cuda()
                labels = labels.cuda()
            number+=1
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs,labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if number % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, number + 1, running_loss / 10))
                running_loss = 0.0

        recognitionRate = 0
        totalNumber = 0
        for inputs, labels in validation_iter:
            if (use_gpu):
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = net(inputs)
            recognitionRate += (outputs.argmax(dim=1) == labels).float().sum().item()
            totalNumber += labels.shape[0]
        recognitionRate = 1.0 * recognitionRate / totalNumber
        print('The recognition rate is %f' % recognitionRate)
        if recognitionRate>maxRecognitionRate:
            torch.save(net.state_dict(), 'model.pkl')
    print('Finished Training')

    net = LeNet()  # 先初始化一个模型，这边的 Model() 指代你的 pytorch 模型
    net.load_state_dict(torch.load('model.pkl', map_location='cpu'))  # 再加载模型参数
    batch_size = 200
    testingdataset = datasets.ImageFolder(root='E:/course-code/meachine-learning/dataset/MNIST/test', transform=transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]))
    test_iter = data.DataLoader(testingdataset, batch_size=batch_size, shuffle=False)
    recognitionRate = 0
    totalNumber = 0
    for inputs, labels in test_iter:
        outputs = net(inputs)
        recognitionRate += (outputs.argmax(dim=1) == labels).float().sum().item()
        totalNumber += labels.shape[0]

    recognitionRate = 1.0*recognitionRate/totalNumber
    print('The recognition rate is %f' % recognitionRate)
