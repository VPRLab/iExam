# Yang Xu

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QDialog
from PyQt5.QtCore import pyqtSignal, QThread, QRegExp
# from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon, QImage, QPixmap, QTextCursor, QFont, QColor, QRegExpValidator
import sys

import cv2
import threading
from datetime import datetime
import multiprocessing
multiprocessing.freeze_support()  # for prevent exe keep opening itself 
import os
import faceClassify  # test face in one frame
import torchTrain  # train by torch
import preProcess  # OCR post-processing
import evaluation  # draw graph
from shutil import rmtree
import numpy as np
import pandas as pd
from configparser import ConfigParser
from PIL import Image
import torchvision.transforms as transforms
from torch.nn.functional import softmax
import torch
from tqdm import tqdm
from ui_faceRecognition import Ui_faceRfecognition
from ui_format import Ui_format



class ProcessThread(QThread):
    process_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, dataset_path, name_lst, viewInfo, frames_dict):
        super().__init__()
        self.dataset = dataset_path
        self.name_lst = name_lst
        self.viewInfo = viewInfo
        self.frames_dict = frames_dict

    def run(self):
        try:
            tmp_dict = {}  # OCR already recognized name in one second
            for num_frame, frame in self.frames_dict.items():
                image, tmp_dict = faceClassify.catchFaceAndClassify(self.dataset, self.name_lst, frame, num_frame, self.viewInfo, tmp_dict)
                self.process_pixmap_signal.emit(image)
        except Exception as e:
            print("error:", e)
            pass

class VideoClassifyThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    classify_finish_signal = pyqtSignal(int)

    def __init__(self, video_path, dataset_path, name_lst, viewInfo):
        super().__init__()
        self.video = video_path
        self.dataset = dataset_path
        self.name_lst = name_lst
        self.viewInfo = viewInfo
        self.threads = {}

    def run(self):
        cap = cv2.VideoCapture(self.video)
        self.viewInfo['Width'] = int(cap.get(3))
        self.viewInfo['Height'] = int(cap.get(4))
        num_frame = 1
        num_threads = 6
        frames_lst = []  # frames_lst=[{}, {}, {}]
        self.viewInfo['fps'] = int(cap.get(5))
        stream = tqdm(total=int(cap.get(7)))
        print('num threads:', num_threads)
        time = datetime.now().strftime('[%d/%m/%Y %H:%M:%S]')
        print(time)
        for i in range(num_threads):
            frames_lst.append({})
        while True:
            ret, cv_img = cap.read()
            # print('the number of captured frame: ', num_frame)
            stream.update(1)
            stream.set_description('Classify Process')
            if ret:
                try:
                    for i in range(num_threads):
                        if num_frame % num_threads == i:
                            frames_lst[i][str(num_frame)] = cv_img
                    if num_frame % (self.viewInfo['fps'] * self.viewInfo['ocr_period'] * num_threads) == 0:
                        # print(len(frames_lst[0]))
                        self.multithread_classify(frames_lst)
                        frames_lst.clear()
                        for i in range(num_threads):
                            frames_lst.append({})

                except Exception as e:
                    print("error:", e)
                    pass
            else:
                self.multithread_classify(frames_lst)  # if frames_lst store some images, but not full
                cap.release()
                break
            num_frame += 1
            # cv2.waitKey(1)
        stream.close()
        time = datetime.now().strftime('[%d/%m/%Y %H:%M:%S]')
        print(time)

        # preprocess part
        preProcess.OCRprocessing(self.dataset, self.name_lst)
        if len(os.listdir(self.dataset)) < 10:
            self.classify_finish_signal.emit(-1)
            return
        elif len(os.listdir(self.dataset)) >= 10:
            self.classify_finish_signal.emit(1)
            return

    def multithread_classify(self, frames_lst):
        try:
            num_threads = len(frames_lst)
            for i in range(num_threads):
                self.threads[i] = ProcessThread(self.dataset, self.name_lst, self.viewInfo, frames_lst[i])
                self.threads[i].process_pixmap_signal.connect(self.emit_image)
                self.threads[i].start()
            for i in range(num_threads):
                self.threads[i].quit()
                self.threads[i].wait()

        except Exception as e:
            print("error:", e)
            pass

    def emit_image(self, img):
        """emit image as signal to main class"""
        self.change_pixmap_signal.emit(img)


class VideoTrainThread(QThread):
    # training part signal
    train_successful_signal = pyqtSignal(dict)
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset = dataset_path

    def run(self):
        time = datetime.now().strftime('[%d/%m/%Y %H:%M:%S]')
        print(time)
        net_path, name_lst, net = torchTrain.trainandsave(self.dataset, epoches=10)
        print(net_path)
        time = datetime.now().strftime('[%d/%m/%Y %H:%M:%S]')
        print(time)
        info_dict = {'net_path': net_path, 'name_lst': name_lst, 'net': net}
        self.train_successful_signal.emit(info_dict)

class VideoTestThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    log_queue = pyqtSignal(str)
    test_finish_signal = pyqtSignal(int)
    def __init__(self, video_path, name_lst, net_path, viewInfo, net):
        super().__init__()
        self.video = video_path
        self.name_lst = name_lst
        self.net_path = net_path
        self.viewInfo = viewInfo
        self.threads = {}
        self.studyCollection = {}
        self.namedict = {}
        self.num_threads = 6
        self.count = 0  # for combine all thread study records
        self.net = net
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # load net
        self.net.load_state_dict(torch.load(self.net_path))
        # self.net.load_state_dict(torch.load(self.net_path, map_location='cpu'))
        self.net.to(self.DEVICE)
        self.net.eval()

    def run(self):
        cap = cv2.VideoCapture(self.video)
        self.viewInfo['Width'] = int(cap.get(3))
        self.viewInfo['Height'] = int(cap.get(4))
        fps = int(cap.get(5))
        self.viewInfo['fps'] = fps
        time_slot = int(fps * self.viewInfo.get('study_period'))
        # for multiprocessing
        frames_lst = [{} for i in range(self.num_threads)]  # frames_lst=[{}, {}, {}]
        num_frame = 1
        stream = tqdm(total=int(cap.get(7)))
        for name in self.name_lst:
            self.studyCollection[name] = 0
            self.namedict[name] = [sys.maxsize, 0]   # a large number for after replace

        time = datetime.now().strftime('[%d/%m/%Y %H:%M:%S]')
        print(time)
        while True:
            ret, cv_img = cap.read()  # read one frame
            # print('the number of captured frame: ', num_frame)
            stream.update(1)
            stream.set_description('Test Process')

            if ret:
                for i in range(self.num_threads):
                    if num_frame % self.num_threads == i:
                        frames_lst[i][str(num_frame)] = cv_img
                if num_frame % (self.viewInfo['fps'] * self.viewInfo['study_period'] * self.num_threads) == 0:
                    # print(len(frames_lst[0]))
                    self.multithread_test(frames_lst, time_slot, num_frame)
                    frames_lst.clear()
                    for i in range(self.num_threads):
                        frames_lst.append({})  # reset

                    time = datetime.now().strftime('[%d/%m/%Y %H:%M:%S]')
                    print(time)

            else:
                cap.release()
                break

            num_frame += 1
            cv2.waitKey(1)
        stream.close()
        self.write_feedback()
        time = datetime.now().strftime('[%d/%m/%Y %H:%M:%S]')
        print(time)
        self.log_queue.emit('Success: faceRecognition')
        self.test_finish_signal.emit(1)
        return


    def multithread_test(self, frames_lst, time_slot, end_frame_num):
        try:
            for i in range(self.num_threads):
                self.threads[i] = TestProcessThread(self.name_lst, self.viewInfo, self.net, frames_lst[i], time_slot, self.DEVICE, end_frame_num)
                self.threads[i].recognize_pixmap_signal.connect(self.emit_image)
                self.threads[i].namedict_signal.connect(self.threads_namedict)
                self.threads[i].studyCollection_signal.connect(self.emit_studyCollection)
                self.threads[i].start()
            for i in range(self.num_threads):
                self.threads[i].quit()
                self.threads[i].wait()

        except Exception as e:
            print("error:", e)
            pass

    def emit_image(self, img):
        """emit image as signal to main class"""
        self.change_pixmap_signal.emit(img)

    def threads_namedict(self, namedict):
        try:
            for name in self.name_lst:
                if self.namedict[name][0] > int(namedict[name][0]):
                    self.namedict[name][0] = int(namedict[name][0])
                self.namedict[name][1] = self.namedict[name][1] + int(namedict[name][1])
        except Exception as e:
            print("namedict error:", e)
            pass

    def emit_studyCollection(self, studyCollection, time_slot, end_frame):
        self.count = self.count + 1
        for name in self.name_lst:
            self.studyCollection[name] = self.studyCollection[name] + studyCollection[name]
        if self.count == 6:
            self.count = 0  # reset count
            log_tmp = []  # people not recognized in list

            for k in self.studyCollection.keys():
                if self.studyCollection[k] == 0:
                    log_tmp.append(k)
                    log_tmp.append('\n')

            fs = open('systemLog.txt', 'a')
            num_frame = int(end_frame)
            if log_tmp:  # if log_tmp is not empty
                log_tmp.pop()  # remove the last \n
                line = 'The following people have not be recognized from ' + str(num_frame // int(self.viewInfo.get('fps')) - int(self.num_threads)+1) + \
                       's to ' + str(num_frame // int(self.viewInfo.get('fps'))) + 's:\n' + "".join(log_tmp)
                self.log_queue.emit(line)
                fs.write(line + '\n')
            else:
                line = 'all students can be recognized from ' + str((num_frame - time_slot) // int(self.viewInfo.get('fps'))* int(self.num_threads)) + \
                       's to ' + str(num_frame // int(self.viewInfo.get('fps'))) + 's'
                self.log_queue.emit(line)
                fs.write(line + '\n')
            fs.close()

            for name in self.name_lst:
                self.studyCollection[name] = 0

    def write_feedback(self):
        f = open('feedback.txt', 'a')
        for k, v in self.namedict.items():
            if self.namedict[k][0] == sys.maxsize:
                line = str(k) + ' has not recognized in this video'
                f.write(line)
                f.write('\n')
                continue
            line = str(k) + ' first detected at ' + str(
                self.namedict[k][0]) + ' frames,' + ' total detect times: ' + str(self.namedict[k][1])
            f.write(line)
            f.write('\n')

        f.close()


class TestProcessThread(QThread):
    recognize_pixmap_signal = pyqtSignal(np.ndarray)
    studyCollection_signal = pyqtSignal(dict, int, int)
    namedict_signal = pyqtSignal(dict)

    def __init__(self, name_lst, viewInfo, net, frames_dict, time_slot, DEVICE, end_frame_num):
        super().__init__()
        self.name_lst = name_lst
        self.viewInfo = viewInfo
        self.net = net
        self.frames_dict = frames_dict
        self.time_slot = time_slot
        self.DEVICE = DEVICE
        self.end_frame_num = end_frame_num


    def run(self):
        try:
            # two methods for get face
           # self.haar_recognition()
           self.dnn_recognition()

        except Exception as e:
            print("error:", e)
            pass

    def haar_recognition(self):
        tmp_dict = {}  # CNN already recognized name in one second
        namedict = {}  # store the time of first recognition, and total recognition time
        studyCollection = {}  # store the recognized time in a time slot

        classes = self.name_lst
        classfier = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
        row = self.viewInfo.get('Row')
        column = self.viewInfo.get('Column')
        clip_width = int(self.viewInfo.get('Width') / row)  # 256
        clip_height = int(self.viewInfo.get('Height') / column)  # 144
        fps = self.viewInfo.get('fps')
        recognize_period = self.viewInfo.get('recognize_period')
        for name in self.name_lst:
            studyCollection[name] = 0
            namedict[name] = [sys.maxsize, 0]

        for num_frame, frame in self.frames_dict.items():

            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if int(num_frame) % int(fps * recognize_period) == 1:  # every recognize period reset tmp_dict
                tmp_dict.clear()
                # print('clear tmp dict')

            face_rects = classfier.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(8, 8))
            if len(face_rects) > 0:
                for face_rect in face_rects:
                    x, y, w, h = face_rect
                    face_row = int(x / clip_width)
                    face_col = int(y / clip_height)
                    if w > clip_width or h > clip_height:  # avoid capture error
                        continue
                    tmp_row = int((x + w) / clip_width)
                    tmp_col = int((y + h) / clip_height)
                    if tmp_row != face_row or tmp_col != face_col:  # avoid capture error
                        continue

                    label = -1  # initialize label, avoid use last one
                    if (str(face_row), str(face_col)) in tmp_dict.keys():
                        label = tmp_dict[(str(face_row), str(face_col))]
                        cv2.putText(frame, classes[label], (clip_width*face_row+30, clip_height*face_col+30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # label name
                    else:
                        image = grey[y - 10:y + h + 10, x - 10:x + w + 10]
                        # opencv to PIL: BGR2RGB
                        PIL_image = self.cv2pil(image)
                        if PIL_image is None:
                            continue
                        # using model to recognize
                        label = self.predict_model(PIL_image)
                        if label != -1:
                            # cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 0, 255), 1)    # draw rectangle
                            cv2.putText(frame, classes[label], (clip_width*face_row+30, clip_height*face_col+30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # label name
                            tmp_dict[(str(face_row), str(face_col))] = label
                        else:
                            continue
                        if label != -1:
                            if namedict[classes[label]][0] > int(num_frame):
                                namedict[classes[label]][0] = int(num_frame)
                            namedict[classes[label]][1] += 1
                            # get the time of this student appear in a time slot
                            studyCollection[classes[label]] += 1

            self.recognize_pixmap_signal.emit(frame)

        self.namedict_signal.emit(namedict)
        self.studyCollection_signal.emit(studyCollection, self.time_slot, self.end_frame_num)

    def dnn_recognition(self):
        tmp_dict = {}  # CNN already recognized name in one second
        namedict = {}  # store the time of first recognition, and total recognition time
        studyCollection = {}  # store the recognized time in a time slot

        classes = self.name_lst
        detector = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
        row = self.viewInfo.get('Row')
        column = self.viewInfo.get('Column')
        clip_width = int(self.viewInfo.get('Width') / row)  # 256
        clip_height = int(self.viewInfo.get('Height') / column)  # 144
        fps = self.viewInfo.get('fps')
        recognize_period = self.viewInfo.get('recognize_period')

        for name in self.name_lst:
            studyCollection[name] = 0
            namedict[name] = [sys.maxsize, 0]

        for num_frame, frame in self.frames_dict.items():

            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if int(num_frame) % int(fps * recognize_period) == 1:  # every recognize period reset tmp_dict
                tmp_dict.clear()
                # print('clear tmp dict')

            base_img = frame.copy()
            original_size = frame.shape
            target_size = (300, 300)
            image = cv2.resize(frame, target_size)
            aspect_ratio_x = (original_size[1] / target_size[1])
            aspect_ratio_y = (original_size[0] / target_size[0])
            imageBlob = cv2.dnn.blobFromImage(image=image)
            detector.setInput(imageBlob)
            detections = detector.forward()
            detections_df = pd.DataFrame(detections[0][0], columns=["img_id", "is_face", "confidence", "left", "top", "right", "bottom"])
            detections_df = detections_df[detections_df['is_face'] == 1]  # 0: background, 1: face
            detections_df = detections_df[detections_df['confidence'] >= 0.15]

            try:
                for i, instance in detections_df.iterrows():
                    # print(instance)
                    left = int(instance["left"] * 300)
                    bottom = int(instance["bottom"] * 300)
                    right = int(instance["right"] * 300)
                    top = int(instance["top"] * 300)

                    detected_face = image[top:bottom, left:right]
                    predict_face = base_img[(int(top * aspect_ratio_y)):(int(bottom * aspect_ratio_y)), (int(left * aspect_ratio_x)):(int(right * aspect_ratio_x))]  # need to change grey -> base_img

                    if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:
                        face_row = int(int(left * aspect_ratio_x) / clip_width)
                        face_col = int(int(top * aspect_ratio_y) / clip_height)
                        if ((bottom - top) * aspect_ratio_y) > clip_width or ((right - left) * aspect_ratio_x) > clip_height:  # avoid capture error
                            continue
                        tmp_row = int(int(right * aspect_ratio_x) / clip_width)
                        tmp_col = int(int(bottom * aspect_ratio_y) / clip_height)
                        if tmp_row != face_row or tmp_col != face_col:  # avoid capture error
                            continue

                        if (str(face_row), str(face_col)) in tmp_dict.keys():
                            historical_name = tmp_dict[(str(face_row), str(face_col))]
                            cv2.putText(base_img, historical_name,(int(clip_width*face_row+30), int(clip_height*face_col+30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                            # cv2.rectangle(base_img, (int(left * aspect_ratio_x), int(top * aspect_ratio_y)), (int(right * aspect_ratio_x), int(bottom * aspect_ratio_y)), (255, 255, 255), 2)  # draw rectangle to main image
                        else:
                            predict_image = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
                            # opencv to PIL: BGR2RGB
                            PIL_image = self.cv2pil(predict_face)
                            if PIL_image is None:
                                continue
                            # using model to recognize
                            label = -1  # initialize label, avoid use last one
                            label = self.predict_model(PIL_image)  # label is idx to name in classes
                            # print('label is: ', label, classes[label])
                            if label != -1:
                                cv2.putText(base_img, classes[label], (int(clip_width*face_row+30), int(clip_height*face_col+30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # label name
                                # cv2.rectangle(base_img, (int(left * aspect_ratio_x), int(top * aspect_ratio_y)), (int(right * aspect_ratio_x), int(bottom * aspect_ratio_y)), (0, 0, 255), 1)  # draw rectangle
                                tmp_dict[(str(face_row), str(face_col))] = classes[label]
                                if namedict[classes[label]][0] > int(num_frame):
                                    namedict[classes[label]][0] = int(num_frame)
                                namedict[classes[label]][1] += 1
                                # get the time of this student appear in a time slot
                                studyCollection[classes[label]] += 1
                            else:
                                continue

            except Exception as e:
                print("frame number:", num_frame, e)
                pass

            self.recognize_pixmap_signal.emit(base_img)

        self.namedict_signal.emit(namedict)
        self.studyCollection_signal.emit(studyCollection, self.time_slot, self.end_frame_num)

    def cv2pil(self, image):
        if image.size != 0:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            return None

    def predict_model(self, image):
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # reszie image to 224*224
            transforms.ToTensor(),  # each pixel to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # image.show()
        image = data_transform(image)  # change PIL image to tensor
        image = image.view(-1, 3, 224, 224)  # change 3-dimensional to 4-dimensional for input

        output = self.net(image.to(self.DEVICE))
        prob = softmax(output[0], dim=0).detach()
        idx = torch.argmax(prob).item()
        # pred = output.max(1, keepdim=True)[1]
        # if pred.item() != idx:
        #     print('no', pred.item())
        # print('output:', prob, 'idx:', idx)
        if prob[idx] > 0.5:  # alexNet:0.98, resnet50:0.05
            return idx
        else:
            return -1


class App(QWidget, Ui_faceRfecognition):

    receiveLogSignal = pyqtSignal(str)  # Log signal

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # loadUi('./faceRecognition.ui', self)

        self.setWindowIcon(QIcon('./icon.png'))
        self.setFixedSize(1280, 656)

        # log system
        self.logQueue = multiprocessing.Queue()  # log queue use multiprocessing
        self.receiveLogSignal.connect(lambda log_received: self.logOutput(log_received))  # signal connect slot
        self.logOutputThread = threading.Thread(target=self.receiveLog, daemon=True)  # multithreading to handle log system and webcam
        self.logOutputThread.start()

        # face collection
        self.isFinishUpload = False
        self.uploadRosterButton.toggled.connect(self.uploadRoster)  # upload all students roster
        self.isFinishFormat = False
        self.clipFormatButton.toggled.connect(self.addInputFormat)  # input view format
        self.isFinishClassify = False
        self.uploadVideoButton.clicked.connect(self.uploadVideo)  # get face data by video and then training
        self.isFinishTrain = False
        self.faceRecognitionButton.clicked.connect(self.faceRecognition)  # select test video
        self.isFinishTest = False
        self.drawButton.clicked.connect(self.plot)  # draw graph

        self.logQueue.put('Step 1: upload roster\n'
                          'Step 2: upload zoom video\n'
                          'Step 3: training dataset\n'
                          'Step 4: invigilating by face recognition')

        self.thread = None  # emit each frame as a signal
        self.video = None  # video path
        self.dataset = None  # dataset name
        self.net_path = None  # model name
        self.name_lst = None  # roster list
        self.formatDialog = formatDialog()
        self.viewInfo = {'Row': '', 'Column': '', 'Width': '', 'Height': '', 'ocr_period': '', 'recognize_period': '', 'study_period': '', 'fps': ''}

        # read config file
        self.config = ConfigParser()
        self.config.read('config.ini', encoding='utf-8-sig')
        self.viewInfo['Row'] = self.config.getint('viewInfo', 'Row')
        self.viewInfo['Column'] = self.config.getint('viewInfo', 'Column')
        self.viewInfo['ocr_period'] = self.config.getint('period', 'ocr_period')
        self.viewInfo['recognize_period'] = self.config.getint('period', 'recognize_period')
        self.viewInfo['study_period'] = self.config.getint('period', 'study_period')

        self.dataset = None
        self.net_path = None
        self.name_lst = None
        self.net = None


    def uploadRoster(self):
        """
        choose student name roster in txt file, each student name should in one line
        """
        self.chooseRosterPath()

        if self.name_lst is not None:
            print('name_lst:', self.name_lst)
            self.uploadRosterButton.setEnabled(False)
            self.logQueue.put('Success: uploadRoster')
            self.isFinishUpload = True
            self.clipFormatButton.setEnabled(True)
        else:
            self.logQueue.put('Warning: not upload roster')

    def addInputFormat(self):
        """
        input zoom gallery mode: 5*5 or 7*7, which will use in frame cut later
        """
        row_num, column_num= self.viewInfo.get('Row'), self.viewInfo.get('Column')
        self.formatDialog.rowLineEdit.setText(str(row_num))
        self.formatDialog.columnLineEdit.setText(str(column_num))
        self.formatDialog.okButton.clicked.connect(self.checkViewInfo)
        self.formatDialog.exec()

    # revise user info
    def checkViewInfo(self):
        if not (self.formatDialog.rowLineEdit.hasAcceptableInput() and
                self.formatDialog.columnLineEdit.hasAcceptableInput()):
            self.formatDialog.msgLabel.setText(
                '<font color=red>Submission Error: check and rewrite！</font>')
        else:
            # get user input
            self.viewInfo['Row'] = int(self.formatDialog.rowLineEdit.text().strip())
            self.viewInfo['Column'] = int(self.formatDialog.columnLineEdit.text().strip())
            self.logQueue.put('Success: Add view format')
            self.isFinishFormat = True
            print('view format: ', self.viewInfo)
            self.formatDialog.close()
            self.clipFormatButton.setEnabled(False)
            self.uploadVideoButton.setEnabled(True)

    def uploadVideo(self):
        """
        get video path
        """
        self.chooseVideo()
        if len(self.video)>0:
            self.uploadVideoButton.setEnabled(False)
            self.logQueue.put('Success: upload training video')
            self.logQueue.put('Start to construct dataset and train model')
            self.logQueue.put('Please waiting ...')
        else:
            self.logQueue.put('Warning: please upload training video')
            return
        # dataset name
        self.dataset = 'marked_image_' + self.video.split('/')[-1].split('.')[0].split('_')[0]

        # remove historical classify folder
        if os.path.exists(self.dataset):
            rmtree(self.dataset)
        # establish newly personal folder
        if not os.path.exists(self.dataset):
            os.makedirs(self.dataset)

        self.thread = VideoClassifyThread(self.video, self.dataset, self.name_lst, self.viewInfo)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.classify_finish_signal.connect(self.successful_classify)
        # self.thread.finished.connect(self.trainData)
        self.thread.start()

    def trainData(self):
        self.thread = VideoTrainThread(self.dataset)
        self.thread.train_successful_signal.connect(self.successful_train)
        self.thread.start()

    def faceRecognition(self):
        # self.logQueue.put('Start face recognition')
        self.chooseVideo()
        if len(self.video)>0:
            self.faceRecognitionButton.setEnabled(False)
            self.logQueue.put('Success: upload testing video')
            self.logQueue.put('Start to testing by face recognition')
        else:
            self.logQueue.put('Warning: please upload testing video')
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print("error:", e)
            pass
        self.thread = VideoTestThread(self.video, self.name_lst, self.net_path, self.viewInfo, self.net)
        self.thread.log_queue.connect(self.send_message)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.test_finish_signal.connect(self.successful_test)
        self.thread.start()

    def plot(self):
        feedback_name = 'feedback.txt'
        log_name = 'systemLog.txt'
        evaluation.drawGraph(feedback_name, log_name)

    def chooseRosterPath(self):
        fileName, filetype = QFileDialog.getOpenFileName(self, "choose roster", os.getcwd(), "Text Files (*.txt)")

        if len(fileName)>0:  # path is not None, then read roster content
            roster = open(fileName, "r")
            self.name_lst = tuple(roster.read().splitlines())

    def chooseVideo(self):
        self.video, filetype = QFileDialog.getOpenFileName(self, "choose video", os.getcwd(), "All Files (*)")

    def send_message(self, message):
        self.logQueue.put(message)

    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.faceDetectCaptureLabel.setPixmap(qt_img)
        self.faceDetectCaptureLabel.setScaledContents(True)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # p = convert_to_Qt_format.scaled(1280, 656, Qt.KeepAspectRatio)
        return QPixmap.fromImage(convert_to_Qt_format)

    # Receive Log
    def receiveLog(self):
        while True:
            try:
                data = self.logQueue.get()  # remove and return an item from queue
                if data:
                    self.receiveLogSignal.emit(data)  # emit signal
                else:
                    continue
            except Exception:
                pass

    # Log Output
    def logOutput(self, log_received):
        # get current time
        time = datetime.now().strftime('[%d/%m/%Y %H:%M:%S]')
        # log = time + '\n' + log_received + '\n'
        self.logTextEdit.moveCursor(QTextCursor.End)  # move cursor to the end
        if 'waiting' in log_received:
            self.logTextEdit.insertPlainText(time + '\n')
            self.logTextEdit.setFontWeight(QFont.Bold)  # change font style to bold
            self.logTextEdit.insertPlainText(log_received)  # insert the text
            self.logTextEdit.setFontWeight(QFont.Normal)  # change font style back to normal
            self.logTextEdit.insertPlainText('\n')
        elif 'Warning' in log_received:
            self.logTextEdit.insertPlainText(time + '\n')
            self.logTextEdit.setTextColor(QColor(255,0,0))  # change font color to red
            self.logTextEdit.insertPlainText(log_received)  # insert the text
            self.logTextEdit.setTextColor(QColor(0,0,0))  # change font color back to black
            self.logTextEdit.insertPlainText('\n')
        else:
            self.logTextEdit.insertPlainText(time + '\n' + log_received + '\n')  # insert the text
        self.logTextEdit.ensureCursorVisible()  # by scrolling text make cursor visible

    def successful_classify(self, switch):
        if switch == -1:
            self.uploadVideoButton.setEnabled(True)
            rmtree(self.dataset)  # remove the dataset which contain too small classes
            self.logQueue.put('Warning: dataset has small number of classes, please reupload a longer training video')
            self.dataset = None
        elif switch == 1:
            self.faceDetectCaptureLabel.setText('<html><head/><body><p><span style=" color:#ff0000;">Zoom Video Window</span></p></body></html>')
            self.logQueue.put('Finished construct dataset, now start training')
            self.trainData()

    def successful_train(self, info_dict):
        if info_dict != {}:
            self.logQueue.put('Success: trainData')
            self.net_path = info_dict['net_path']
            self.name_lst = info_dict['name_lst']
            self.net = info_dict['net']
            self.isFinishTrain = True
            self.faceRecognitionButton.setEnabled(True)

    def successful_test(self, switch):
        if switch == 1:
            self.faceDetectCaptureLabel.setText('<html><head/><body><p><span style=" color:#ff0000;">Zoom Video Window</span></p></body></html>')
            self.drawButton.setEnabled(True)


class formatDialog(QDialog, Ui_format):
    def __init__(self):
        super(formatDialog, self).__init__()
        self.setupUi(self)
        # loadUi('./format.ui', self)
        self.setFixedSize(512, 248)

        # restrict the input
        regx = QRegExp('^[0-9]$')
        row_validator = QRegExpValidator(regx, self.rowLineEdit)
        self.rowLineEdit.setValidator(row_validator)
        column_validator = QRegExpValidator(regx, self.columnLineEdit)
        self.columnLineEdit.setValidator(column_validator)


if __name__=="__main__":
    if getattr(sys, 'frozen', False):
        os.environ['JOBLIB_MULTIPROCESSING'] = '0'  # for prevent exe keep opening itself
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())