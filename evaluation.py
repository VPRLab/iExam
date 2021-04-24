import multiprocessing
import sys
import threading
from datetime import datetime

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PyQt5.uic import loadUi
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QWidget, QApplication
import os
import torchTrain


class Evaluation(QWidget):

    def __init__(self):
        super(Evaluation, self).__init__()
        loadUi('./evluation.ui', self)
        self.setFixedSize(512, 293)

        # face collection
        self.isFinishEvaluate = False
        self.evaluateButton.clicked.connect(self.test)  # upload test dataset
        self.isFinishDraw = False
        self.drawButton.clicked.connect(self.drawGraph)  # draw graph

        self.dataset = None  # dataset name
        self.net_path = None  # model name
        self.name_lst = None  # roster list
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def chooseDataset(self):
        self.dataset = QFileDialog.getExistingDirectory(self, "choose test dataset", os.getcwd(), QFileDialog.ShowDirsOnly)
        print('dataset: ', self.dataset)


    def loadtestdata(self):
        testset = torchvision.datasets.ImageFolder(self.dataset, transform=transforms.Compose([
            transforms.Resize((32, 32)),  # resize image (h,w)
            transforms.ToTensor()]))
        print('classes: ', testset.classes)  # all classes in dataset
        self.name_lst = tuple(testset.classes)
        testloader = torch.utils.data.DataLoader(testset, batch_size=25, shuffle=True, num_workers=2)
        return testloader

    def reload_net(self):
        print('len of list: ', len(self.name_lst))
        model = torchTrain.Net(len(self.name_lst))
        model.load_state_dict(torch.load('net_params_5min_of90min.pth'))
        model.eval()
        return model


    def test(self):
        self.chooseDataset()
        if self.dataset is '':
            return
        testloader = self.loadtestdata()
        print('test data loaded')
        net = self.reload_net()
        print('net loaded')
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                output = net(x)
                test_loss += F.nll_loss(output, y, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(y.view_as(pred)).sum().item()
                print('correct:', correct)

        test_loss /= len(testloader.dataset)
        print('test loss={:.4f}, accuracy={:.4f}'.format(test_loss, float(correct) / len(testloader.dataset)))


    def drawGraph(self):
        # draw a bar chart to show all reg times
        file = open("feedback_25min_of30min.txt", 'r')
        lines = file.readlines()
        all_reg_times = {}
        for idx, line in enumerate(lines):
            line = line[:-1]
            if 'first detected at' in line:
                tmp1 = line.split(' first')
                tmp2 = line.split('times: ')
                all_reg_times[tmp1[0]] = int(tmp2[1])
            elif 'has not recognized' in line:
                tmp1 = line.split(' has')
                all_reg_times[tmp1[0]] = 0

        lst_data = sorted(all_reg_times.items(), key=lambda items: items[1], reverse=True)  # descending data of reg time
        print(len(lst_data))
        print(lst_data)
        lst_data_name = []
        lst_data_num = []
        for i in range(len(lst_data)):
            lst_data_num.append(lst_data[i][1])
            lst_data_name.append((lst_data[i][0]))

        print('name: ', lst_data_name)
        print('num: ', lst_data_num)

        plt.figure(figsize=(10, 6))  # set figure size
        plt.tick_params(axis='y', labelsize=7)  # set ylabel size
        plt.xlabel('Recognition Times')
        plt.barh(lst_data_name, lst_data_num)
        plt.title('conclusion of recognition times for 25min test video')
        plt.savefig('feedback_conclusion.png')
        plt.show()

        # draw a graph according present distribution
        file = open("systemLog_25min_of30min.txt", 'r')
        lines = file.readlines()
        all_log_time = {}
        all_log_name = {}
        tmp = 0
        for idx, line in enumerate(lines):
            line = line[:-1]
            # print(idx, ' line: ', line)
            if 'following people have not be recognized' in line:
                tmp1 = line.split('from ')[1]
                tmp2 = tmp1.split(' to ')
                tmp2[0] = tmp2[0][:-1]
                tmp2[1] = tmp2[1][:-2]
                all_log_time[idx] = tmp2
                all_log_name[idx] = []
                tmp = idx  # tmp store idx
            elif 'all students can be recognized' in line:
                tmp1 = line.split('from ')[1]
                tmp2 = tmp1.split(' to ')
                tmp2[0] = tmp2[0][:-1]
                tmp2[1] = tmp2[1][:-2]
                all_log_time[idx] = tmp2
                all_log_name[idx] = []
            else:
                all_log_name[tmp].append(line)

        print(all_log_time)
        print(len(all_log_time))
        print(all_log_name)

        time_slot = None
        color_lst = []
        for key in all_log_time.keys():
            tmp_lst = []
            for i in range(len(lst_data_name)):
                tmp_lst.append('b')
            time_slot = int(all_log_time[key][1]) - int(all_log_time[key][0])
            for name in all_log_name[key]:
                idx = lst_data_name.index(name)
                tmp_lst[idx] = 'r'
            color_lst.append(tmp_lst)

        print('color: ', color_lst)

        x = []
        for i in range(len(lst_data_name)):
            x.append(time_slot)

        bars = []
        for i in range(len(lst_data_name)):
            bars.append(x)

        ind = np.arange(len(bars))
        bar_categories = lst_data_name
        bar_width = 0.5

        plt.figure(figsize=(12, 6))  # set figure size
        plt.tick_params(axis='x', labelsize=7)  # set xlabel size
        plt.xticks(rotation=35)
        plt.ylabel('Time(seconds)')

        plt.bar(ind, x, width=bar_width, color=color_lst[0])
        plt.bar(ind, x, bottom=x, width=bar_width, color=color_lst[1])
        bottom = np.add(x, x)
        for i in range(len(all_log_time) - 2):
            plt.bar(ind, x, bottom=bottom.tolist(), width=bar_width, color=color_lst[i + 2])
            bottom = np.add(x, bottom)

        plt.xticks(ind, bar_categories)
        plt.xlabel("Stacked Bar Plot")
        plt.title('recognition distribution')
        plt.savefig('systemLog_conclusion.png')
        plt.show()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = Evaluation()
    window.show()
    sys.exit(app.exec())

    # marked_image_5min_of10min
    # test loss = -23.6462, accuracy = 0.9980

    # marked_image_5min_of30min
    # test loss = -26.5499, accuracy = 0.9972

    # marked_image_5min_of90min
    # test loss = -25.6272, accuracy = 0.9955


