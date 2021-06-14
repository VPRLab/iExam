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
import cv2
import difflib
import platform
import pytesseract
from torchTrain import Net
from PIL import Image
from collections import defaultdict





class Evaluation(QWidget):

    def __init__(self):
        super(Evaluation, self).__init__()
        loadUi('./evluation.ui', self)
        self.setFixedSize(512, 293)

        # face collection
        self.isFinishEvaluate = False
        self.evaluateButton.clicked.connect(self.test)  # upload test dataset
        self.trueEvaluateButton.clicked.connect(self.testByComparison)  # compare the text to test accuracy
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
        model.load_state_dict(torch.load('net_params_5min_of10min.pth'))
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

    # find the longest consecutive sequence
    def findLongest(self, nums):
        if nums == []:
            return 0
        nums.sort()
        maxlen = 1
        curlen = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 1:
                curlen += 1
                maxlen = max(maxlen, curlen)
            else:
                maxlen = max(maxlen, curlen)
                curlen = 1
        return maxlen

    def drawGraph(self):
        # draw a bar chart to show all reg times
        file = open("feedback_90min_of90min.txt", 'r')
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
        plt.title('conclusion of recognition times for 90min test video')
        # plt.savefig('feedback_conclusion_90min.png')
        # plt.show()

        # draw a graph according present distribution
        file = open("systemLog_90min_of90min.txt", 'r')
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
        plt.title('recognition distribution for 90min test video')
        # plt.savefig('systemLog_conclusion_90min.png')
        plt.show()


        # draw a diagram with priority (max disappear in continuous time)

        priority_dict = {}
        result = {}
        for name in lst_data_name:
            priority_dict[name] = []
            result[name] = 0
        tmp_idx = 1

        for key in all_log_name.keys():
            for item in all_log_name[key]:
                priority_dict[item].append(tmp_idx)
            tmp_idx += 1
        print('priority_dict: ', priority_dict)

        for key in priority_dict.keys():
            result[key] = self.findLongest(priority_dict[key])
        sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        print(sorted_result)

        for i in range(len(sorted_result)):
            sorted_result[i] = list(sorted_result[i])
            sorted_result[i][1] = time_slot * sorted_result[i][1]
        print(sorted_result)

        plt.figure(figsize=(12, 6))  # set figure size
        plt.tick_params(axis='x', labelsize=7)  # set xlabel size
        plt.xticks(rotation=35)
        plt.ylabel('Time(seconds)')
        plt.title('Consecutive disappear time for 90min test video')

        tmp_name = []
        tmp_data = []
        for i in range(len(sorted_result)):
            tmp_name.append(sorted_result[i][0])
            tmp_data.append(sorted_result[i][1])
        plt.bar(tmp_name, tmp_data)

        plt.savefig('consecutive_disappear_90min.png')
        plt.show()

    def testByComparison(self):
        video = '25min_testVideo_of30min.mp4'
        name_lst = ('ZhijingBAO', 'RuikaiCAI', 'KexinCAO', 'QianqianCAO', 'PakKwanCHAN', 'YuMingCHAN',
                    'GuozhengCHEN', 'HanweiCHEN', 'JiahuangCHEN', 'JiaxianCHEN', 'QijieCHEN', 'YirunCHEN',
                    'YuqinCHENG', 'TszKuiCHOW', 'LiujiaDU', 'BowenFAN', 'RouwenGE', 'RuiGUO', 'DuoHAN',
                    'YouyangHAN', 'BailinHE', 'JiayiHOU', 'BingHU', 'BingyuanHUANG', 'HoNamLAI', 'DonghaoLI',
                    'QingboLI', 'SiqinLI', 'SiruiLI', 'ZiyiLI', 'HaoweiLIU', 'JinzhangLIU', 'KaihangLIU', 'LeiLIU',
                    'ZiwenLU', 'KuanLV', 'ChenghaoLYU', 'MingshengMA', 'SuweiSUN', 'YuanTIAN', 'HiuYanTONG', 'AnboWANG',
                    'HaoWANG', 'RunzeWANG', 'YuchuanWANG', 'YanWU', 'RuochenXIE', 'MengYIN', 'ZijingYIN', 'NgokTingYIP',
                    'FanyuanZENG', 'QidongZHAI', 'LiZHANG', 'YalingZHANG', 'ZiyaoZHANG', 'ZicongZHENG', 'ShengtongZHU',
                    'YifanZHU', 'YimingZOU')

        net_path = 'net_params_5min_of30min.pth'
        classes = ['BailinHE', 'BingyuanHUANG', 'DonghaoLI', 'DuoHAN', 'FanyuanZENG', 'GuozhengCHEN', 'HanweiCHEN',
                   'HaoWANG', 'HaoweiLIU', 'JiahuangCHEN', 'JiayiHOU', 'KaihangLIU', 'LeiLIU', 'LiZHANG', 'LiujiaDU',
                   'MengYIN', 'MingshengMA', 'NgokTingYIP', 'QidongZHAI', 'RuikaiCAI', 'RunzeWANG', 'ShengtongZHU',
                   'SuweiSUN', 'TszKuiCHOW', 'YalingZHANG', 'YirunCHEN', 'YuMingCHAN', 'YuchuanWANG', 'ZhijingBAO',
                   'ZicongZHENG', 'ZiyaoZHANG', 'ZiyiLI']


        cap = cv2.VideoCapture(video)
        num_frame = 1
        time = datetime.now().strftime('[%d/%m/%Y %H:%M:%S]')
        print(time)
        namedict = defaultdict(list)
        classfier = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")
        if platform.system() == 'Windows':
            pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        elif platform.system() == 'Darwin':
            pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
        row = 5
        column = 5
        clip_width = 256
        clip_height = 144
        while True:
            ret, cv_img = cap.read()
            if ret:
                print('the number of captured frame: ', num_frame)
                print(namedict)
                grey = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)  # change to greystyle
                faceRects = classfier.detectMultiScale(cv_img, 1.1, 5, minSize=(8, 8))  # objects are returned as a list of rectangles.
                if len(faceRects) > 0:
                    for faceRect in faceRects:
                        x, y, w, h = faceRect

                        for i in range(column):
                            for j in range(row):
                                if clip_width * j <= x <= clip_width * (j + 1) and clip_height * i <= y <= clip_height * (
                                        i + 1):
                                    cropped = grey[clip_height * (i + 1) - 32:clip_height * (i + 1),
                                              clip_width * j:clip_width * j + 120]  # decrease length
                                    cropped = cv2.resize(cropped, None, fx=1.2, fy=1.2)
                                    # if pixel greyscale>185, set this pixel=255, preprocess the character image to get good quality for OCR
                                    ret, thresh1 = cv2.threshold(cropped, 185, 255, cv2.THRESH_TOZERO)
                                    text = pytesseract.image_to_string(thresh1)  # OCR
                                    text = ''.join([char for char in text if
                                                    char.isalpha()])  # remove the character like '\n',' ','\0Xc'
                                    # print('before text is:', text)
                                    text = self.string_comparison(text, name_lst)

                                    image = grey[y - 10:y + h + 10, x - 10:x + w + 10]
                                    # opencv to PIL: BGR2RGB
                                    PIL_image = self.cv2pil(image)
                                    if PIL_image is None:
                                        continue
                                    # using model to recognize
                                    label = self.predict_model(PIL_image, net_path, len(classes))
                                    if classes[label] == text:
                                        if not namedict[classes[label]]:
                                            namedict[classes[label]].append(1)  # [1, 1] first one is correct time. second is total time
                                            namedict[classes[label]].append(1)
                                        else:
                                            namedict[classes[label]][0] += 1
                                            namedict[classes[label]][1] += 1
                                    else:
                                        if not namedict[classes[label]]:
                                            namedict[classes[label]].append(0)  # [1, 1] first one is correct time. second is total time
                                            namedict[classes[label]].append(1)
                                        else:
                                            namedict[classes[label]][1] += 1

            if not ret:
                cap.release()
                break
            num_frame += 1
            # cv2.waitKey(1)
        time = datetime.now().strftime('[%d/%m/%Y %H:%M:%S]')
        print(time)
        print(namedict)

    def string_comparison(self, text, name_lst):  # get rid of small difference of OCR for the same character
        simlar_str = text
        lst = difflib.get_close_matches(text, name_lst, n=1, cutoff=0.75)
        if lst:
            return lst[0]
        else:
            return simlar_str

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize(32),  # reszie image to 32*32
            transforms.CenterCrop(32),  # center crop 32*32
            transforms.ToTensor()  # each pixel to tensor
        ])

    def cv2pil(self, image):
        if image.size != 0:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            return None

    def predict_model(self, image, net_path, class_num):

        data_transform = self.get_transform()
        image = data_transform(image)  # change PIL image to tensor
        image = image.view(-1, 3, 32, 32)
        net = Net(class_num)
        # net = torch.load(net_path)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(DEVICE)
        # load net
        net.load_state_dict(torch.load(net_path))
        output = net(image.to(DEVICE))
        # get the maximum
        pred = output.max(1, keepdim=True)[1]
        return pred.item()






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


