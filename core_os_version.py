# Yang Xu

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon, QImage, QPixmap, QTextCursor, QFont, QColor


import sys
import cv2
import threading
from datetime import datetime
import multiprocessing
import os
import faceClassify  # test face in one frame
import torchTrain  # train by torch
import preProcess  # OCR post processing
import faceTestUsingTorch  # face recognition
import shutil
from collections import defaultdict
import numpy as np


class VideoClassifyThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, video_path, dataset_path, name_lst):
        super().__init__()
        self.video = video_path
        self.dataset = dataset_path
        self.name_lst = name_lst

    def run(self):

        cap = cv2.VideoCapture(self.video)
        num_frame = 1
        while True:
            ret, cv_img = cap.read()
            if ret:
                print('the number of captured frame: ', num_frame)
                image = faceClassify.catchFaceAndClassify(self.dataset, self.name_lst, cv_img, num_frame)
                self.change_pixmap_signal.emit(image)
            else:
                cap.release()
                break
            num_frame += 1
            cv2.waitKey(1)


class VideoTestThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    log_queue = pyqtSignal(str)
    def __init__(self, video_path, dataset_path, name_lst, net_path):
        super().__init__()
        self.video = video_path
        self.dataset = dataset_path
        self.name_lst = name_lst
        self.net_path = net_path

    def run(self):
        cap = cv2.VideoCapture(self.video)
        fps = cap.get(5)
        num_frame = 1
        studyCollection = {}
        namedict = defaultdict(list)
        time_slot = 20 * int(fps)  # time slot(frame=20seconds*fps) for checking whether student is checked,
        for name in self.name_lst:
            studyCollection[name] = 0

        while cap.isOpened():
            ret, cv_img = cap.read()  # read one frame
            if not ret:
                cap.release()
                break

            print('the number of captured frame: ' + str(num_frame))
            image, namedict, studyCollection = faceTestUsingTorch.recognize(self.name_lst, cv_img, namedict, num_frame,
                                                                            self.net_path, studyCollection, time_slot)
            self.change_pixmap_signal.emit(image)

            if num_frame % time_slot == 0:  # every one time slot check
                log_tmp = []  # people not recognized in list

                for k in studyCollection.keys():
                    if studyCollection[k] == 0:
                        log_tmp.append(k)
                        log_tmp.append('\n')

                fs = open('systemLog.txt', 'a')
                if log_tmp:  # if log_tmp is not empty
                    log_tmp.pop()  # remove the last \n
                    line = 'The following people have not be recognized from ' + str((num_frame - time_slot)//int(fps)) + \
                           's to ' + str(num_frame//int(fps)) + 's:\n' + "".join(log_tmp)
                    self.log_queue.emit(line)
                    # self.logQueue.put(line)
                    fs.write(line + '\n')
                else:
                    line = 'all students can be recognized from ' + str((num_frame - time_slot)//int(fps)) + \
                           's to ' + str(num_frame//int(fps)) + 's'
                    self.log_queue.emit(line)
                    fs.write(line + '\n')
                fs.close()

            num_frame += 1
            cv2.waitKey(1)

        self.log_queue.emit('Success: faceRecognition')

        f = open('feedback.txt', 'a')
        for k, v in namedict.items():
            line = str(k)+' first detected at '+str(namedict[k][0])+' frames,'+' total detect times: '+str(namedict[k][1])
            f.write(line)
            f.write('\n')

        for name in self.name_lst:
            if name not in namedict:
                line = name + 'has not recognized in this video'
                f.write(line)
                f.write('\n')

        f.close()





class App(QWidget):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    log_queue = pyqtSignal(str)
    receiveLogSignal = pyqtSignal(str)  # Log signal

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Qt live label demo")
        loadUi('./faceRecognition.ui', self)

        self.setWindowIcon(QIcon('./icon.png'))
        self.setFixedSize(1280, 656)

        # log system
        self.logQueue = multiprocessing.Queue()  # log queue use multiprocessing
        self.receiveLogSignal.connect(lambda log_received: self.logOutput(log_received))  # signal connect slot
        self.logOutputThread = threading.Thread(target=self.receiveLog, daemon=True)  # multithreading to handle log system and webcam
        self.logOutputThread.start()


        # self.cap = cv2.VideoCapture()


        # face collection
        self.isFinishUpload = False
        self.uploadRosterButton.toggled.connect(self.uploadRoster)  # upload all students roster
        self.isFinishClassify = False
        self.uploadVideoButton.clicked.connect(self.uploadVideo)  # get face data by video
        self.isFinishTrain = False
        self.isFinishTest = False
        self.faceRecognitionButton.clicked.connect(self.faceRecognition)  # training classified images



        self.logQueue.put('Step 1: upload roster\n'
                          'Step 2: upload zoom video\n'
                          'Step 3: training dataset\n'
                          'Step 4: take attendance by face recognition')

        self.thread = None  # emit each frame as a signal
        self.video = None  # video path
        self.dataset = None  # dataset name
        self.net_path = None  # model name
        self.name_lst = None  # roster list


        # self.dataset = 'marked_image_10minCopy'
        # self.net_path = 'net_params_10minCopy.pkl'
        # self.name_lst = ('BailinHE', 'BingHU', 'BowenFAN', 'ChenghaoLYU', 'HanweiCHEN',
        #                  'JiahuangCHEN', 'LiZHANG', 'LiujiaDU', 'PakKwanCHAN', 'QijieCHEN',
        #                  'RouwenGE', 'RuiGUO', 'RunzeWANG', 'RuochenXIE', 'SiqinLI',
        #                  'SiruiLI', 'TszKuiCHOW', 'YanWU', 'YimingZOU', 'YuMingCHAN',
        #                  'YuanTIAN', 'YuchuanWANG', 'ZiwenLU', 'ZiyaoZHANG')


        # cascade classifier
        # self.faceCascade = cv2.CascadeClassifier('./lbpcascades/lbpcascade_frontalface.xml')
        # self.faceCascade = cv2.CascadeClassifier('./lbpcascades/lbpcascade_profileface.xml')
        self.faceCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml') #ok
        # self.eyeCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
        # self.faceCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt.xml')
        # self.faceCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt_tree.xml')
        # self.faceCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt2.xml') # fast haar
        # self.eyeCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_lefteye_2splits.xml')
        # self.eyeCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_righteye_2splits.xml')


    def uploadRoster(self):
        self.chooseRosterPath()

        if self.name_lst is not None:
            print('name_lst:', self.name_lst)
            self.uploadRosterButton.setEnabled(False)
            self.logQueue.put('Success: uploadRoster')
            self.isFinishUpload = True
            self.uploadVideoButton.setEnabled(True)
        else:
            self.logQueue.put('Warning: not upload roster')

    def uploadVideo(self):
        # capture from web cam
        # path = self.video
        self.chooseVideo()
        if len(self.video)>0:
            self.uploadVideoButton.setEnabled(False)
            self.logQueue.put('Success: upload training video')
            self.logQueue.put('Start to construct dataset and train model')
            self.logQueue.put('Please waiting ...')
        else:
            self.logQueue.put('Warning: please upload training video')
            return

        self.dataset = 'marked_image_' + self.video.split('/')[-1].split('_')[0]

        # remove historical folder
        if os.path.exists(self.dataset):
            shutil.rmtree(self.dataset)
        # establish newly personal folder
        if not os.path.exists(self.dataset):
            os.makedirs(self.dataset)


        self.thread = VideoClassifyThread(self.video, self.dataset, self.name_lst)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.finished.connect(self.trainData)
        self.thread.start()




    def trainData(self):
        self.faceDetectCaptureLabel.setText('<html><head/><body><p><span style=" color:#ff0000;">Zoom Video Window</span></p></body></html>')
        self.isFinishClassify = True
        preProcess.OCRprocessing(self.dataset, self.name_lst)


        if len(os.listdir(self.dataset)) < 10:
            self.uploadVideoButton.setEnabled(True)
            shutil.rmtree(self.dataset)  # remove the dataset which contain too small classes
            self.logQueue.put('Warning: dataset has small number of classes, please reupload a longer training video')
            self.dataset = None
            # self.num_frame = 1  # reset frame number
            return


        self.net_path, self.name_lst = torchTrain.trainandsave(self.dataset)
        self.logQueue.put('Success: trainData')
        self.isFinishTrain = True
        self.faceRecognitionButton.setEnabled(True)


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

        self.thread = VideoTestThread(self.video, self.dataset, self.name_lst, self.net_path)

        self.thread.log_queue.connect(self.send_message)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        # self.cap = cv2.VideoCapture(self.video)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # fps = self.cap.get(5)  # get the video fps
        # self.num_frame = 1


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


if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())