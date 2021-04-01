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


def loadtraindata():
    path = chooseDataset()
    filename = path.split('/')[-1].split('_')[-1]
    print('path:', path)
    print('filename:', filename)
    trainset = torchvision.datasets.ImageFolder(path,
                transform=transforms.Compose([transforms.Resize((32, 32)),  # resize image (h,w)
                transforms.CenterCrop(32), transforms.ToTensor()]))
    print('classes: ', trainset.classes)  # all classes in dataset
    print(trainset)
    # batch_size: number of iteration in each time
    # shuffle: whether random sort in each time
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    return trainloader, filename


class Net(nn.Module):  # define net, which extends torch.nn.Module
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # convolution layer
        self.pool = nn.MaxPool2d(2, 2)  # pooling layer
        self.conv2 = nn.Conv2d(6, 16, 5)  # convolution layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # fully connected layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 24)  # output is 24, 24 is the number of class in dataset

    def forward(self, x):  # feed forward

        x = self.pool(F.relu(self.conv1(x)))  # F is torch.nn.functional
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # .view( ) is a method tensor, which automatically change tensor size but elements number not change

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def trainandsave():
    trainloader, filename = loadtraindata()

    net = Net()
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
    torch.save(net, 'net_'+filename+'.pkl')  # save structure and parameter
    torch.save(net.state_dict(), 'net_params_'+filename+'.pkl')  # only save parameter



# test ==========================================

classes_3min = ('BailinHE', 'BingyuanHUANG', 'DonghaoLI', 'DuoHAN', 'FanyuanZENG',
           'HaoWANG', 'HaoweiLIU', 'JiahuangCHEN', 'KaihangLIU', 'LeiLIU',
           'MengYIN', 'MingshengMA', 'NgokTingYIP', 'QidongZHAI', 'RuikaiCAI',
           'RunzeWANG', 'ShengtongZHU', 'YalingZHANG', 'YirunCHEN', 'YuqinCHENG',
           'ZhijingBAO', 'ZiyaoZHANG', 'ZiyiLI')

classes_10min = ('BailinHE', 'BingHU', 'BowenFAN', 'ChenghaoLYUk', 'HanweiCHEN',
                 'JiahuangCHEN', 'LiZHANG', 'LiujiaDU', 'PakKwanCHAN', 'QijieCHEN',
                 'RouwenGE', 'RuiGUO', 'RunzeWANG', 'RuochenXie', 'SiqinLI',
                 'SiruiLI', 'TszKuiCHOW', 'YanWU', 'YimingZOU', 'YuMingCHAN',
                 'YuanTIAN', 'YuchuanWANG', 'ZiwenLU', 'ZiyaoZHANG')

classes = classes_10min

def loadtestdata():
    path = chooseDataset()
    testset = torchvision.datasets.ImageFolder(path, transform=transforms.Compose([
            transforms.Resize((32, 32)),  # resize image (h,w)
            transforms.ToTensor()]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=25, shuffle=True, num_workers=2)
    return testloader

def reload_net():
    trainednet = torch.load('net_3minCopy.pkl')
    return trainednet

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()
    print('plt show')

def test():
    testloader = loadtestdata()
    print('test data loaded')
    net = reload_net()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = net(x)
            test_loss += F.nll_loss(output, y, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
            print('correct:', correct)

    test_loss /= len(testloader.dataset)
    print('test loss={:.4f}, accuracy={:.4f}'.format(test_loss, float(correct) / len(testloader.dataset)))

    # dataiter = iter(testloader)
    # images, labels = dataiter.next()  #
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # # imshow(torchvision.utils.make_grid(images, nrow=5))  # nrow is the size of image in one row
    # print('GroundTruth: ', " ".join('%5s' % classes[labels[j]] for j in range(25)))  # print 25
    #
    # outputs = net(Variable(images))
    # _, predicted = torch.max(outputs.data, 1)
    # print('Predicted: ', " ".join('%5s' % classes[predicted[j]] for j in range(25)))



if __name__ == '__main__':
    # trainandsave()
    test()

    # marked_image_3minCopy
    # test_loss = -24.3998, accuracy = 0.9909
