import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter.filedialog

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def chooseDataset():
    root = tkinter.Tk()  # create a Tkinter.Tk() instance
    root.withdraw()  # hide Tkinter.Tk() instance
    datasets = tkinter.filedialog.askdirectory(title=u'choose file')  # choose dataset path
    if datasets == '':
        exit()
    else:
        return datasets


def loadtraindata(path):
    # path = chooseDataset()
    filename = path.split('/')[-1].split('_')[-1]
    # print('path:', path)
    # print('filename:', filename)
    trainset = torchvision.datasets.ImageFolder(path,
                transform=transforms.Compose([transforms.Resize((32, 32)),  # resize image (h,w)
                transforms.CenterCrop(32), transforms.ToTensor()]))
    print('classes: ', trainset.classes)  # all classes in dataset
    print('number of classes: ', len(trainset.classes))

    print(trainset)
    # batch_size: number of iteration in each time
    # shuffle: whether random sort in each time
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    return trainloader, filename, tuple(trainset.classes)


class Net(nn.Module):  # define net, which extends torch.nn.Module
    def __init__(self, class_num):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # convolution layer
        self.pool = nn.MaxPool2d(2, 2)  # pooling layer
        self.conv2 = nn.Conv2d(6, 16, 5)  # convolution layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # fully connected layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, class_num)  # output is class_num, class_num is the number of class in dataset

    def forward(self, x):  # feed forward

        x = self.pool(F.relu(self.conv1(x)))  # F is torch.nn.functional
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # .view( ) is a method tensor, which automatically change tensor size but elements number not change

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def trainandsave(path):
    trainloader, filename, classes = loadtraindata(path)

    net = Net(len(classes))
    net.to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # learning rate=0.001
    criterion = nn.CrossEntropyLoss()  # loss function
    # training part
    for epoch in range(5):  # 5 epoch
        # each epoch train all images, so total train 5 times
        running_loss = 0.0  # loss output, training 200 images will output running_loss
        for i, (inputs, labels) in enumerate(trainloader):
            # wrap them in Variable, Variable will recode all operation for tensors
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # change format to Variable
            optimizer.zero_grad()  # reset gradient to 0, because feed back will add last gradient
            # forward + backward + optimize
            outputs = net(inputs)  # put input to cnn net
            loss = criterion(outputs, labels)  # calculate loss
            loss.backward()  # loss feed backward
            optimizer.step()  # refresh all parameter
            running_loss += loss.data  # loss accumulation
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))  # mean loss for each 200 images
                running_loss = 0.0  # reset

    print('Finished Training')
    # save net
    # torch.save(net, 'net_'+filename+'.pkl')  # save structure and parameter
    torch.save(net.state_dict(), 'net_params_'+filename+'.pkl')  # only save parameter

    return 'net_params_'+filename+'.pkl', classes


if __name__ == '__main__':
    trainandsave()

