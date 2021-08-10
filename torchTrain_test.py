import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time
import numpy as np
from torchvision import models

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def loadtraindata(path):
    # path = chooseDataset()
    filename = path.split('/')[-1].split('_')[-1]
    # print('path:', path)
    # print('filename:', filename)
    dataset = torchvision.datasets.ImageFolder(path,
                                               transform=transforms.Compose(
                                                   [transforms.Resize((224, 224)),  # resize image (h,w)
                                                    transforms.CenterCrop(224), transforms.ToTensor()]))
    print('classes: ', dataset.classes)  # all classes in dataset
    print('number of classes: ', len(dataset.classes))
    print(dataset)

    train_size = int(len(dataset) * 0.8)
    validate_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - validate_size
    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validate_size,
                                                                                            test_size])

    # batch_size: number of iteration in each time
    # shuffle: whether random sort in each time
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=8,
                                               pin_memory=True)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=20, shuffle=False, num_workers=8,
                                                  pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=2,
                                              pin_memory=True)

    return train_loader, validate_loader, test_loader, filename, tuple(dataset.classes)


class Net(nn.Module):  # define net, which extends torch.nn.Module
    def __init__(self, class_num):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # convolution layer
        self.conv2 = nn.Conv2d(6, 16, 5)  # convolution layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # fully connected layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, class_num)  # output is class_num, class_num is the number of class in dataset
        self.pool = nn.MaxPool2d(2, 2)  # pooling layer

    def forward(self, x):  # feed forward
        x = self.pool(F.relu(self.conv1(x)))  # F is torch.nn.functional
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(x.shape[0], -1)  # .view( ) is a reshape operation, which automatically change tensor size but elements number not change
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def trainandsave(path):
    train_loader, validate_loader, test_loader, filename, classes = loadtraindata(path)
    # network 1:
    # net = Net(len(classes))
    # network 2:
    # net = models.alexnet(pretrained=True)
    # net.features = nn.Sequential(
    #     nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
    #     nn.ReLU(inplace=True),
    #     nn.MaxPool2d(kernel_size=3, stride=2),
    #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
    #     nn.ReLU(inplace=True),
    #     nn.MaxPool2d(kernel_size=3, stride=2),
    #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
    #     nn.ReLU(inplace=True),
    #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
    #     nn.ReLU(inplace=True),
    #     # nn.Conv2d(256, 256, kernel_size=3, padding=1),
    #     # nn.ReLU(inplace=True),
    #     nn.MaxPool2d(kernel_size=3, stride=2),
    # )
    # net.classifier = nn.Sequential(
    #     nn.Dropout(),
    #     nn.Linear(256 * 6 * 6, 4096),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(),
    #     nn.Linear(4096, 4096),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(4096, len(classes)),
    # )
    # network 3:
    net = models.resnet18(pretrained=True)
    net.fc = nn.Linear(512, len(classes))

    print(net)
    net.to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # learning rate=0.008
    criterion = nn.CrossEntropyLoss()  # loss function
    previous_validate_loss = 1000 # enough bigger
    # training
    train_loss_set = []
    val_loss_set = []
    train_correct_set = []
    val_correct_set = []
    try:
        for epoch in range(25):  # 3 epoch
            # each epoch train all images, so total train 5 times
            net.train()
            running_loss = 0.0  # loss output, training 200 images will output running_loss
            tr_loss = []
            train_correct = 0
            train_total = 0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()  # reset gradient to 0, because feed back will add last gradient
                # forward + backward + optimize
                outputs = net(inputs)  # put input to cnn net
                loss = criterion(outputs, labels)  # calculate loss
                tr_loss.append(loss.data.cpu())
                pred = outputs.max(1, keepdim=True)[1]
                train_correct += pred.eq(labels.view_as(pred)).sum().item()
                loss.backward()  # loss feed backward
                optimizer.step()  # refresh all parameter
                running_loss += loss.data  # loss accumulation
                train_total += 20

                if i % 200 == 199:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 200))  # mean loss for each 200 images
                    running_loss = 0.0  # reset

            train_loss = np.mean(tr_loss)
            train_loss_set.append(train_loss)
            train_correct_set.append(float(train_correct)/train_total)
            print('tr_loss:',tr_loss)
            print('train loss:', train_loss)

            net.eval()
            val_loss = []
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(validate_loader):
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = net(inputs)  # put input to cnn net
                    loss = criterion(outputs, labels)  # calculate loss
                    # print("loss data:", loss.data)
                    val_loss.append(loss.data.cpu())  # loss accumulation
                    pred = outputs.max(1, keepdim=True)[1]
                    val_correct += pred.eq(labels.view_as(pred)).sum().item()
                    val_total += 20

            validate_loss = np.mean(val_loss)
            val_loss_set.append(validate_loss)
            val_correct_set.append(float(val_correct) / val_total)
            print('val_loss:', val_loss)
            print('validate loss:', validate_loss)

        print('train_loss_set:', train_loss_set)
        print('val_loss_set:', val_loss_set)
        print('train_accuracy_set:', train_correct_set)
        print('val_accuracy_set:', val_correct_set)

        # test_loss = 0
        # correct = 0
        # total = 0.0
        # with torch.no_grad():
        #     for x, y in test_loader:
        #         x, y = x.to(DEVICE), y.to(DEVICE)
        #         output = net(x)
        #
        #         loss = criterion(output, y)  # calculate loss
        #         test_loss += loss.data  # loss accumulation
        #
        #         # test_loss += F.nll_loss(output, y, reduction='sum').item()
        #         # print(test_loss)
        #         pred = output.max(1, keepdim=True)[1]
        #         correct += pred.eq(y.view_as(pred)).sum().item()
        #         total += 200
        #         # print('correct:', correct, ' total number: ', total, ' accuracy: ', float(correct) / total)
        #
        # test_loss /= len(test_loader.dataset)
        # print('test loss={:.4f}, accuracy={:.4f}'.format(test_loss, float(correct) / len(test_loader.dataset)))

    except Exception as e:
        print("error:", e)
        pass
    print('Finished Training')
    torch.save(net.state_dict(), 'net_params_' + filename + '_alexnettest.pth')  # only save parameter

    return 'net_params_' + filename + '.pth', classes


if __name__ == "__main__":
    path = 'marked_image_5min'
    before = time.asctime(time.localtime(time.time()))
    print(before)
    trainandsave(path)
    after = time.asctime(time.localtime(time.time()))
    print(after)
