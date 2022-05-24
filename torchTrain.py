import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torchvision import models, utils
from torchvision import datasets
from torchvision import transforms
from collections import defaultdict
from tqdm import tqdm
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
import os
from sklearn.metrics import classification_report, confusion_matrix
# from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import warnings
import random

import cv2
from PIL import Image

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(42)
    torch.cuda.empty_cache()
    # memory_summary = torch.cuda.memory_summary(device='cuda', abbreviated=False)
    # print(memory_summary)
else:
    DEVICE = torch.device('cpu')
    torch.manual_seed(42)

np.random.seed(100)


def calculate_accuracy(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {'val': 0, 'count': 0, 'avg': 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric['val'] += val
        metric['count'] += 1
        metric['avg'] = metric['val'] / metric['count']

    def __str__(self):
        return ' | '.join(
            [
                '{metric_name}: {avg:.{float_precision}f}'.format(
                    metric_name=metric_name, avg=metric['avg'], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def train(train_loader, net, criterion, optimizer, epoch, scaler, writer=None):
    metric_monitor = MetricMonitor()
    net.train()
    stream = tqdm(train_loader)
    train_accuracy = []
    train_loss = []
    for batch_index, (images, labels) in enumerate(stream, start=1):
        images = images.to(DEVICE, non_blocking=True)  # tensor size [batch_size, 3, img_w, img_h]
        labels = labels.to(DEVICE, non_blocking=True)  # tensor size [batch_size, ]
        # dataset visualization using tensorboard
        # img_grid = utils.make_grid(images)
        # writer.add_image('train_images', img_grid)
        # writer.add_graph(net, images)

        optimizer.zero_grad()
        with autocast():
            outputs = net(images)  # tensor size [batch_size, len(classes)]
            loss = criterion(outputs, labels)  # calculate loss
            accuracy = calculate_accuracy(outputs, labels)
            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('Accuracy', accuracy.item())
            train_accuracy.append(accuracy.item())
            train_loss.append(loss.item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        stream.set_description(
            'Epoch: {epoch}. Train.      {metric_monitor}'.format(epoch=epoch, metric_monitor=metric_monitor)
        )
    return train_loss, train_accuracy, scaler


def validate(val_loader, net, criterion, epoch):
    metric_monitor = MetricMonitor()
    net.eval()
    stream = tqdm(val_loader)
    val_accuracy = []
    val_loss = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(stream, start=1):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            outputs = net(images)
            loss = criterion(outputs, labels)
            accuracy = calculate_accuracy(outputs, labels)

            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('Accuracy', accuracy.item())
            val_accuracy.append(accuracy.item())
            val_loss.append(loss.item())
            stream.set_description(
                'Epoch: {epoch}. Validation. {metric_monitor}'.format(epoch=epoch, metric_monitor=metric_monitor)
            )
    return val_loss, val_accuracy


def test(test_loader, net, idx2class):
    metric_monitor = MetricMonitor()
    net.eval()
    stream = tqdm(test_loader)
    y_pred_list = []
    y_true_list = []
    test_accuracy = []
    with torch.no_grad():
        for x_batch, y_batch in stream:
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)
            y_test_pred = net(x_batch)
            _, y_pred_tag = torch.max(y_test_pred, dim=1)
            y_pred_list.append(y_pred_tag.cpu().numpy().tolist())   # list shape: [_, batch_size]
            y_true_list.append(y_batch.cpu().numpy().tolist())
            accuracy = calculate_accuracy(y_test_pred, y_batch)
            metric_monitor.update('Accuracy', accuracy.item())
            test_accuracy.append(accuracy.item())
            stream.set_description(
                'Test.  {metric_monitor}'.format(metric_monitor=metric_monitor)
            )

    y_pred_array = []
    y_true_array = []
    for i in range(len(y_pred_list)):
        y_pred_array += y_pred_list[i]  # 1-dimension
        y_true_array += y_true_list[i]
    # print('y_pred_list: ', y_pred_array)
    # print('y_true_list: ', y_true_array)
    target_names = [v for k, v in idx2class.items()]

    print('classification_report')
    print(classification_report(y_true_array, y_pred_array, target_names=target_names))
    print('confusion_matrix')
    print(confusion_matrix(y_true_array, y_pred_array))
    # confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true_array, y_pred_array)).rename(columns=idx2class, index=idx2class)
    # fig, ax = plt.subplots(figsize=(20, 18))
    # sns.heatmap(confusion_matrix_df, annot=True, ax=ax)
    # plt.savefig('Heatmap.eps', dpi=600, format='eps')
    # plt.show()
    return test_accuracy


def get_class_distribution(dataset, idx2class):
    count_dict = {k: 0 for k, v in dataset.class_to_idx.items()}
    for _, label_id in dataset:
        label = idx2class[label_id]
        count_dict[label] += 1
    print('count_dict: ', count_dict)
    return count_dict


def get_class_distribution_loaders(dataloader_obj, dataset_obj, idx2class):
    count_dict = {k: 0 for k, v in dataset_obj.class_to_idx.items()}
    if dataloader_obj.batch_size == 1:
        for _, label_id in dataloader_obj:
            y_idx = label_id.item()
            y_lbl = idx2class[y_idx]
            count_dict[str(y_lbl)] += 1
    else:
        for _, label_id in dataloader_obj:
            for idx in label_id:
                y_idx = idx.item()
                y_lbl = idx2class[y_idx]
                count_dict[str(y_lbl)] += 1
    return count_dict


def plot_from_dict(dict_obj, plot_title, legend_switch, **kwargs):
    data = pd.DataFrame.from_dict([dict_obj]).melt().rename(columns={'variable': 'stu_name', 'value': 'img_num'})
    if legend_switch:
        plot = sns.barplot(data=data, x='stu_name', y='img_num', hue='stu_name', dodge=False, **kwargs)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    else:
        plot = sns.barplot(data=data, x='stu_name', y='img_num', dodge=False, **kwargs)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=45)
    plot.set_title(plot_title, fontsize=20)
    plot.set_ylabel('image number', fontsize=20)
    return plot


def loadtraindata(path):
    filename = path.split('/')[-1].split('_')[-1]
    print('path:', path)
    print('filename:', filename)

    image_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-90, 90)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    # method 1 for construct training dataset and validate dataset
    # draw whole dataset class distribution
    if os.path.exists(path + '/.DS_Store'):
        os.remove(path + '/.DS_Store')
    dataset = datasets.ImageFolder(path, transform=image_transforms['train'])
    idx2class = {v: k for k, v in dataset.class_to_idx.items()}
    # plt.figure(figsize=(18, 8))
    # plot_from_dict(get_class_distribution(dataset, idx2class), plot_title='Entire Dataset', legend_switch=1)
    # plt.savefig('Entire Dataset Distribution.eps', dpi=600, format='eps')
    print('classes: ', dataset.classes)  # all classes in dataset
    print('number of classes: ', len(dataset.classes), dataset.class_to_idx)
    print('dataset: ', type(dataset), dataset)
    dataset_size = len(dataset)
    dataset_indices = list(range(dataset_size))
    np.random.shuffle(dataset_indices)
    val_split_index = int(np.floor(0.7 * dataset_size))  # train 70%, val 20%, test 10%
    test_split_index = int(np.floor(0.9 * dataset_size))
    train_idx, val_idx, test_idx = dataset_indices[:val_split_index], dataset_indices[val_split_index:test_split_index], dataset_indices[test_split_index:]
    print('len: ', len(train_idx), len(val_idx), len(test_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    # batch_size: number of iteration in each time
    # when use SubsetRandomSampler cannot use shuffle, shuffle: whether random sort in each time
    train_loader = DataLoader(dataset=dataset, batch_size=20, shuffle=False, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset=dataset, batch_size=20, shuffle=False, sampler=val_sampler, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset=dataset, batch_size=20, shuffle=False, sampler=test_sampler, num_workers=8, pin_memory=True)


    '''
    use new test dataset to test model accuracy, need to delete finally
    '''
    # test_dataset = 'dataset_test'
    # test_dataset = datasets.ImageFolder(test_dataset, transform=image_transforms['test'])
    # dataset_size = len(test_dataset)
    # dataset_indices = list(range(dataset_size))
    # np.random.shuffle(dataset_indices)
    # test_split_index = int(np.floor(0.1 * dataset_size))
    # test_idx = dataset_indices[:test_split_index]
    # test_sampler = SubsetRandomSampler(test_idx)
    # idx2class_test = {v: k for k, v in test_dataset.class_to_idx.items()}
    # test_loader = DataLoader(dataset=test_dataset, batch_size=20, shuffle=False, sampler=test_sampler, num_workers=8, pin_memory=True)
    

    print('len of training set: ', len(train_loader), train_loader)
    print('len of validation set: ', len(val_loader), val_loader)
    print('len of test set: ', len(test_loader), test_loader)

    # draw train, val, test dataset class distribution
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8))
    # plot_from_dict(get_class_distribution_loaders(train_loader, dataset, idx2class), plot_title='Train Set', legend_switch=0, ax=axes[0])
    # plot_from_dict(get_class_distribution_loaders(val_loader, dataset, idx2class), plot_title='Val Set', legend_switch=1, ax=axes[1])
    # plt.savefig('Train and Validate Distribution.eps', dpi=600, format='eps')
    # plt.figure(figsize=(18, 8))
    # plot_from_dict(get_class_distribution_loaders(test_loader, dataset, idx2class), plot_title='Test Set', legend_switch=1)
    # plt.savefig('Test Distribution.eps', dpi=600, format='eps')
    # plt.show()

    # method 2 for construct training dataset and validate dataset
    # draw whole dataset class distribution
    # if os.path.exists(path + '/.DS_Store'):
    #     os.remove(path + '/.DS_Store')
    # dataset = datasets.ImageFolder(path)
    # idx2class = {v: k for k, v in dataset.class_to_idx.items()}
    # plt.figure(figsize=(18, 8))
    # plot_from_dict(get_class_distribution(dataset, idx2class), plot_title='Entire Dataset', legend_switch=1)
    # plt.savefig('Entire Dataset Distribution.eps', dpi=600, format='eps')
    # stu_class = os.listdir(path)
    # train_dataset = path + '/train'
    # val_dataset = path + '/validate'
    # test_dataset = path + '/test'
    # if os.path.exists(train_dataset):
    #     shutil.rmtree(train_dataset)
    # if os.path.exists(val_dataset):
    #     shutil.rmtree(val_dataset)
    # if os.path.exists(test_dataset):
    #     shutil.rmtree(test_dataset)
    # os.makedirs(train_dataset)
    # os.makedirs(val_dataset)
    # os.makedirs(test_dataset)
    # for i in range(len(stu_class)):
    #     file_path_one_stu = path + '/' + stu_class[i]
    #     file_list_one_stu = os.listdir(file_path_one_stu)
    #
    #     file_list_one_stu = sorted(file_list_one_stu)
    #     np.random.shuffle(file_list_one_stu)
    #     # print(file_list_one_stu)
    #     val_split_index = int(np.floor(0.7 * len(file_list_one_stu)))  # train 70%, val 20%, test 10%
    #     test_split_index = int(np.floor(0.9 * len(file_list_one_stu)))
    #     train_idx, val_idx, test_idx = file_list_one_stu[:val_split_index], file_list_one_stu[val_split_index:test_split_index], file_list_one_stu[test_split_index:]
    #     os.makedirs(train_dataset + '/' + stu_class[i])
    #     os.makedirs(val_dataset + '/' + stu_class[i])
    #     os.makedirs(test_dataset + '/' + stu_class[i])
    #     for file in os.listdir(file_path_one_stu):
    #         if file in train_idx:
    #             shutil.move(file_path_one_stu + '/' + file, train_dataset + '/' + stu_class[i])
    #         elif file in val_idx:
    #             shutil.move(file_path_one_stu + '/' + file, val_dataset + '/' + stu_class[i])
    #         else:
    #             shutil.move(file_path_one_stu + '/' + file, test_dataset + '/' + stu_class[i])
    #     shutil.rmtree(file_path_one_stu)
    #
    # train_dataset = datasets.ImageFolder(train_dataset, transform=image_transforms['train'])
    # val_dataset = datasets.ImageFolder(val_dataset, transform=image_transforms['train'])
    # test_dataset = datasets.ImageFolder(test_dataset, transform=image_transforms['test'])
    # train_loader = DataLoader(dataset=train_dataset, batch_size=20, shuffle=True, num_workers=8, pin_memory=True)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=20, shuffle=True, num_workers=8, pin_memory=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=20, shuffle=True, num_workers=8, pin_memory=True)
    #
    # print('len of training set: ', len(train_loader), train_loader)
    # print('len of validation set: ', len(val_loader), val_loader)
    # print('len of test set: ', len(test_loader), test_loader)
    #
    # # draw train, val, test dataset class distribution
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8))
    # plot_from_dict(get_class_distribution(train_dataset, idx2class), plot_title='Train Set', legend_switch=0, ax=axes[0])
    # plot_from_dict(get_class_distribution(val_dataset, idx2class), plot_title='Val Set', legend_switch=1, ax=axes[1])
    # plt.savefig('Train and Validate Distribution.eps', dpi=600, format='eps')
    # plt.figure(figsize=(18, 8))
    # plot_from_dict(get_class_distribution(test_dataset, idx2class), plot_title='Test Set', legend_switch=1)
    # plt.savefig('Test Distribution.eps', dpi=600, format='eps')
    # plt.show()

    return train_loader, val_loader, test_loader, filename, tuple(dataset.classes), idx2class


def loadtestdata(test_dataset='dataset_test'):
    filename = test_dataset.split('/')[-1].split('_')[-1]
    print('path:', test_dataset)
    print('filename:', filename)

    image_transforms = {
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    # draw whole dataset class distribution
    if os.path.exists(test_dataset + '/.DS_Store'):
        os.remove(test_dataset + '/.DS_Store')
    dataset = datasets.ImageFolder(test_dataset, transform=image_transforms['test'])
    idx2class = {v: k for k, v in dataset.class_to_idx.items()}
    # plt.figure(figsize=(18, 8))
    # plot_from_dict(get_class_distribution(dataset, idx2class), plot_title='Entire Dataset', legend_switch=1)
    # plt.savefig('Entire Dataset Distribution.eps', dpi=600, format='eps')
    print('classes: ', dataset.classes)  # all classes in dataset
    print('number of classes: ', len(dataset.classes), dataset.class_to_idx)
    print('dataset: ', type(dataset), dataset)

    '''
    use new test dataset to test model accuracy, need to delete finally
    '''
    test_ratio = 1
    test_dataset = datasets.ImageFolder(test_dataset, transform=image_transforms['test'])
    dataset_size = len(test_dataset)
    dataset_indices = list(range(dataset_size))
    np.random.shuffle(dataset_indices)
    test_split_index = int(np.floor(test_ratio * dataset_size))
    test_idx = dataset_indices[:test_split_index]
    # test_sampler = SubsetRandomSampler(test_idx)
    # batch_size: number of iteration in each time
    # when use SubsetRandomSampler cannot use shuffle, shuffle: whether random sort in each time
    test_loader = DataLoader(dataset=test_dataset, batch_size=20, shuffle=False, num_workers=8, pin_memory=True)

    print('len of test set: ', len(test_loader), test_loader)

    return test_loader, filename, tuple(dataset.classes), idx2class


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
        x = x.view(x.shape[0],
                   -1)  # .view( ) is a method tensor, which automatically change tensor size but elements number not change

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def trainandsave(path, epoches):
    train_loader, validate_loader, test_loader, filename, classes, idx2class = loadtraindata(path)
   

    # myNet
    # net = Net(len(classes))

    # alexNet
    # net = models.alexnet(pretrained=True, )
    # net.classifier[6] = nn.Linear(4096, len(classes))

    # googleNet:
    # net = models.googlenet(pretrained=True)
    # net.fc = nn.Linear(1024, len(classes))

    # resNet18
    # net = models.resnet18(pretrained=True, )
    # net.fc = nn.Linear(512 * models.resnet.BasicBlock.expansion, len(classes))

    # resNet50
    net = models.resnet50(pretrained=True, )
    net.fc = nn.Linear(512 * models.resnet.Bottleneck.expansion, len(classes))

    # resNet101
    # net = models.resnet101(pretrained=True, )
    # net.fc = nn.Linear(512 * models.resnet.Bottleneck.expansion, len(classes))

    # resnet152
    # net = models.resnet152(pretrained=True, )
    # net.fc = nn.Linear(512 * models.resnet.Bottleneck.expansion, len(classes))

    # densenet121
    # net = models.densenet121(pretrained=True, )
    # net.classifier = nn.Linear(1024, len(classes))

    # densenet161
    # net = models.densenet161(pretrained=True, )
    # net.classifier = nn.Linear(2208, len(classes))

    # vgg11 CUDA out of memory
    # net = models.vgg11(pretrained=True, )
    # net.classifier[6] = nn.Linear(4096, len(classes))

    # vgg11_bn CUDA out of memory
    # net = models.vgg11_bn(pretrained=True, )
    # net.classifier[6] = nn.Linear(4096, len(classes))

    # mobilenet_v3
    # net = models.mobilenet_v3_small(pretrained=False, num_classes=len(classes), )

    # net = models.mobilenet_v3_small(pretrained=True, )
    # net.classifier[3] = nn.Linear(1024, len(classes))

    # squeezenet1_1
    # net = models.squeezenet1_1(pretrained=True, )
    # net.classifier[1] = nn.Conv2d(512, len(classes), kernel_size=(1,1), stride=(1,1))

    # inception_v3
    # net = models.inception_v3(pretrained=True, aux_logits=True)
    # net.fc = nn.Linear(2048, len(classes))
    # net.AuxLogits = models.inception.InceptionAux(768, len(classes))

    # print(net)
    net.to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # learning rate=0.001
    criterion = nn.CrossEntropyLoss().to(DEVICE)  # loss function
    scaler = GradScaler()  # automatic mixed precision

    # training part
    for epoch in range(epoches):  # 10 epoch
        # each epoch train all images, so total train 10 times
        # writer = SummaryWriter('runs/iExam_20220127_0518_1')
        # train_loss, train_accuracy,scaler = train(train_loader, net, criterion, optimizer, epoch, scaler, writer)
        train_loss, train_accuracy, scaler = train(train_loader, net, criterion, optimizer, epoch, scaler)
        val_loss, val_accuracy = validate(validate_loader, net, criterion, epoch)

        # test_accuracy = test(test_loader, net, idx2class)
        # writer.add_scalar('loss/train', np.asarray(np.mean(train_loss)), epoch)
        # writer.add_scalar('acc/train', np.asarray(np.mean(train_accuracy)), epoch)
        # writer.add_scalar('loss/val', np.asarray(np.mean(val_loss)), epoch)
        # writer.add_scalar('acc/val', np.asarray(np.mean(val_accuracy)), epoch)

        textfile = open('tmp.txt', 'a')
        textfile.write('epoch_{0}_train_loss = ['.format(epoch))
        string = ', '.join(str(item) for item in train_loss)
        textfile.write(string + ']\n')
        textfile.write('epoch_{0}_train_accuracy = ['.format(epoch))
        string = ', '.join(str(item) for item in train_accuracy)
        textfile.write(string + ']\n')
        textfile.write('epoch_{0}_val_loss = ['.format(epoch))
        string = ', '.join(str(item) for item in val_loss)
        textfile.write(string + ']\n')
        textfile.write('epoch_{0}_val_accuracy = ['.format(epoch))
        string = ', '.join(str(item) for item in val_accuracy)
        textfile.write(string + ']\n')
        textfile.close()


    print('Finished Training')
    test_accuracy = test(test_loader, net, idx2class)
    # print('test accuracy:', test_accuracy)
    print('Finished Test')
    # textfile = open('tmp.txt', 'a')
    # textfile.write('Test_accuracy = [')
    # string = ', '.join(str(item) for item in test_accuracy)
    # textfile.write(string + ']\n')
    # textfile.close()

    # save net
    # torch.save(net, 'net_'+filename+'.pkl')  # save structure and parameter
    file_name = 'net_params_resnet50_Pretrain_10epoch_224x224_' + filename + '.pth'
    torch.save(net.state_dict(), file_name)  # only save parameter

    return file_name, classes, net


def another_dataset_test(net_path, test_dataset='dataset_test'):
    test_loader, filename, classes, idx2class = loadtestdata(test_dataset)

    # googleNet:
    # net = models.googlenet(pretrained=True)
    # net.fc = nn.Linear(1024, len(classes))

    # resNet18
    # net = models.resnet18(pretrained=True, )
    # net.fc = nn.Linear(512 * models.resnet.BasicBlock.expansion, len(classes))

    # resNet50
    net = models.resnet50(pretrained=True, )
    net.fc = nn.Linear(512 * models.resnet.Bottleneck.expansion, len(classes))

    # print(net)
    net.load_state_dict(torch.load(net_path))
    net.to(DEVICE)
    test_accuracy = test(test_loader, net, idx2class)
    # print('test accuracy:', test_accuracy)
    print('Finished Test')
    textfile = open('tmp.txt', 'a')
    textfile.write('Test_accuracy = [')
    string = ', '.join(str(item) for item in test_accuracy)
    textfile.write(string + ']\n')
    textfile.close()

def image_test(net_path, classes):
    # googleNet:
    # net = models.googlenet(pretrained=True)
    # net.fc = nn.Linear(1024, len(classes))

    # resNet18
    # net = models.resnet18(pretrained=True, )
    # net.fc = nn.Linear(512 * models.resnet.BasicBlock.expansion, len(classes))

    # resNet50
    net = models.resnet50(pretrained=True, )
    net.fc = nn.Linear(512 * models.resnet.Bottleneck.expansion, len(classes))

    # alexNet
    # net = models.alexnet(pretrained=True, )
    # net.classifier[6] = nn.Linear(4096, len(classes))

    net.load_state_dict(torch.load(net_path))
    net.to(DEVICE)
    net.eval()


    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # reszie image to 224*224
        transforms.ToTensor(),  # each pixel to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_pred_lst = []
    img_pred_score = []
    metric_monitor = MetricMonitor()
    print('classes:',classes)
    stream = tqdm(image_sampler())
    for name in stream:
        image = cv2.imread(name)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # image.show()
        image = data_transform(image)  # change PIL image to tensor
        # print('before:', image.shape, image)
        image = image.view(-1, 3, 224, 224)  # change 3-dimensional to 4-dimensional for input
        # print('after:',image.shape, image)

        output = net(image.to(DEVICE, non_blocking=True))
        y_pred_softmax = torch.log_softmax(output, dim=1)
        _, y_pred_tag = torch.max(y_pred_softmax, dim=1)
        tmp = y_pred_tag.cpu().detach().numpy()
        if classes[tmp[0]] != name.split('\\')[1].split('_')[0]:
            img_pred_score.append(0)
            metric_monitor.update('Accuracy', 0)
        else:
            img_pred_score.append(1)
            metric_monitor.update('Accuracy', 1)
        img_pred_lst.append(classes[tmp[0]])
        stream.set_description(
            'Test.  {metric_monitor}'.format(metric_monitor=metric_monitor)
        )

            # print(y_pred_tag.cpu().detach().numpy())
#     print(img_pred_lst)
    img_compare = zip(img_pred_lst, stream)
    # print('img_compare', tuple(img_compare))
    print('sum: ', sum(img_pred_score), 'len is: ', len(img_pred_score))


def image_sampler(image_path='dataset_fusion'):
    img_path_lst = []
    return_lst = []
    for root, dirs, files in os.walk(image_path):
        for name in files:
            img_path_lst.append(os.path.join(root, name))

    for num in range(len(img_path_lst)//5):
        while True:
            tmp = random.randint(0, len(img_path_lst))
            if img_path_lst[tmp] in return_lst:
                continue
            else:
                return_lst.append(img_path_lst[tmp])
                break
    # print(return_lst)
    return return_lst

if __name__ == '__main__':
    path = 'marked_image_5min'
    before = time.asctime(time.localtime(time.time()))
    print(before)
    # trainandsave(path, epoches=10)
    after = time.asctime(time.localtime(time.time()))
    print(after)