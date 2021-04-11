# Yang Xu

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5.QtCore import pyqtSignal
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

class core(QWidget):
    receiveLogSignal = pyqtSignal(str)  # Log signal

    def __init__(self):
        # load ui and set window size and icon
        super(core, self).__init__()
        loadUi('./faceRecognition.ui', self)
        self.setWindowIcon(QIcon('./icon.png'))
        self.setFixedSize(1280, 656)

        # log system
        self.logQueue = multiprocessing.Queue()  # log queue use multiprocessing
        self.receiveLogSignal.connect(lambda log_received: self.logOutput(log_received))  # signal connect slot
        self.logOutputThread = threading.Thread(target=self.receiveLog, daemon=True)  # multithreading to handle log system and webcam
        self.logOutputThread.start()


        self.cap = cv2.VideoCapture()


        # face collection
        self.isFinishUpload = False
        self.uploadRosterButton.toggled.connect(self.uploadRoster)  # upload all students roster
        self.isFinishClassify = False
        self.uploadVideoButton.clicked.connect(self.uploadVideo)  # get face data by video
        self.isFinishTrain = False
        self.traindataButton.clicked.connect(self.trainData)  # training classified images
        self.isFinishTest = False
        self.faceRecognitionButton.clicked.connect(self.faceRecognition)  # training classified images

        self.logQueue.put('Step 1: upload roster\n'
                          'Step 2: upload zoom video\n'
                          'Step 3: training dataset\n'
                          'Step 4: take attendance by face recognition')

        self.video = None
        self.num_frame = 1
        self.dataset = None
        self.net_path = None
        self.name_lst = None

        # self.traindataButton.setEnabled(True)  # use to test train function, after need to delete


        # self.dataset = 'marked_image_10minCopy'
        # self.net_path = 'net_params_10minCopy.pkl'
        # self.name_lst = ('BailinHE', 'BingHU', 'BowenFAN', 'ChenghaoLYU', 'HanweiCHEN',
        #                  'JiahuangCHEN', 'LiZHANG', 'LiujiaDU', 'PakKwanCHAN', 'QijieCHEN',
        #                  'RouwenGE', 'RuiGUO', 'RunzeWANG', 'RuochenXIE', 'SiqinLI',
        #                  'SiruiLI', 'TszKuiCHOW', 'YanWU', 'YimingZOU', 'YuMingCHAN',
        #                  'YuanTIAN', 'YuchuanWANG', 'ZiwenLU', 'ZiyaoZHANG')


        # cascade classifier
        # self.faceCascade = cv2.CascadeClassifier('../lbpcascades/lbpcascade_frontalface.xml')
        # self.faceCascade = cv2.CascadeClassifier('../lbpcascades/lbpcascade_profileface.xml')
        self.faceCascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml') #ok
        # self.eyeCascade = cv2.CascadeClassifier('../haarcascades/haarcascade_eye.xml')
        # self.faceCascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')
        # self.faceCascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt_tree.xml')
        # self.faceCascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt2.xml') # fast haar
        # self.eyeCascade = cv2.CascadeClassifier('../haarcascades/haarcascade_lefteye_2splits.xml')
        # self.eyeCascade = cv2.CascadeClassifier('../haarcascades/haarcascade_righteye_2splits.xml')

    # upload all students roster
    def uploadRoster(self):
        self.chooseRosterPath()
        print('name_lst:', self.name_lst)
        if self.name_lst is not None:
            self.uploadRosterButton.setEnabled(False)
            self.logQueue.put('Success: uploadRoster')
            self.isFinishUpload = True
            self.uploadVideoButton.setEnabled(True)
        else:
            self.logQueue.put('Warning: not upload roster')

    # test by uploaded video
    def uploadVideo(self):
        self.catchVideo()
        self.faceDetectCaptureLabel.setText('<html><head/><body><p><span style=" color:#ff0000;">Zoom Video Window</span></p></body></html>')

    # training classified images by torch
    def trainData(self):

        # self.dataset = 'marked_image_8min'  # use to test train function, after need to delete

        self.logQueue.put('Start preprocessing')
        self.logQueue.put('Please waiting ...')
        preProcess.OCRprocessing(self.dataset, self.name_lst)
        self.logQueue.put('Finish preprocessing')
        self.logQueue.put('Start training')
        if len(os.listdir(self.dataset)) < 10:
            self.uploadVideoButton.setEnabled(True)
            self.traindataButton.setEnabled(False)
            shutil.rmtree(self.dataset)  # remove the dataset which contain too small classes
            self.logQueue.put('Warning: dataset has small number of classes, please reupload a longer training video')
            return

        self.net_path, self.name_lst = torchTrain.trainandsave(self.dataset)
        self.logQueue.put('Success: trainData')
        self.traindataButton.setEnabled(False)
        self.isFinishTrain = True
        self.faceRecognitionButton.setEnabled(True)

    def faceRecognition(self):
        # self.logQueue.put('Start face recognition')
        self.testVideo()


    def chooseRosterPath(self):
        fileName, filetype = QFileDialog.getOpenFileName(self, "choose roster", os.getcwd(), "Text Files (*.txt)")

        if len(fileName)>0:  # path is not None, then read roster content
            roster = open(fileName, "r")
            self.name_lst = tuple(roster.read().splitlines())

    def chooseVideo(self):
        self.video, filetype = QFileDialog.getOpenFileName(self, "choose video", os.getcwd(), "All Files (*)")

    def catchVideo(self):
        self.chooseVideo()
        if len(self.video)>0:
            self.uploadVideoButton.setEnabled(False)
            self.logQueue.put('Success: upload training video')
            self.logQueue.put('Start to construct dataset')
            self.logQueue.put('Please waiting ...')
        else:
            self.logQueue.put('Warning: please upload training video')
            return
        self.dataset = 'marked_image_' + self.video.split('/')[-1].split('_')[0]
        self.cap = cv2.VideoCapture(self.video)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # remove historical folder
        if os.path.exists(self.dataset):
            shutil.rmtree(self.dataset)
        # establish newly personal folder
        if not os.path.exists(self.dataset):
            os.makedirs(self.dataset)

        while self.cap.isOpened():
            ok, frame = self.cap.read()  # read one frame
            # width, height = cap.get(3), cap.get(4)
            # print('width height:',width,height)
            # property: width 1280, height 720, one frame:40ms  1280/5 = 256 720/5 = 144
            if not ok:
                self.cap.release()
                break

            print('the number of captured frame: ' + str(self.num_frame))
            image = faceClassify.catchFaceAndClassify(self.dataset, self.name_lst, frame, self.num_frame)
            self.displayImage(image)

            self.num_frame += 1
            cv2.waitKey(1)
        self.logQueue.put('Success: finish data collection')
        self.isFinishClassify = True
        self.traindataButton.setEnabled(True)

    def testVideo(self):
        self.chooseVideo()
        if len(self.video)>0:
            self.faceRecognitionButton.setEnabled(False)
            self.logQueue.put('Success: upload testing video')
            self.logQueue.put('Start to testing by face recognition')
        else:
            self.logQueue.put('Warning: please upload testing video')
            return
        self.cap = cv2.VideoCapture(self.video)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        fps = self.cap.get(5)  # get the video fps
        self.num_frame = 1

        studyCollection = {}
        namedict = defaultdict(list)
        time_slot = 20 * int(fps)  # time slot(frame=20seconds*fps) for checking whether student is checked,
        for name in self.name_lst:
            studyCollection[name] = 0

        while self.cap.isOpened():
            ok, frame = self.cap.read()  # read one frame
            if not ok:
                self.cap.release()
                break

            print('the number of captured frame: ' + str(self.num_frame))
            image, namedict, studyCollection = faceTestUsingTorch.recognize(self.name_lst, frame, namedict, self.num_frame,
                                                                            self.net_path, studyCollection, time_slot)
            self.displayImage(image)


            if self.num_frame % time_slot == 0:  # every one time slot check
                log_tmp = []  # people not recognized in list

                for k in studyCollection.keys():
                    if studyCollection[k] == 0:
                        log_tmp.append(k)
                        log_tmp.append('\n')

                fs = open('systemLog.txt', 'a')
                if log_tmp:  # if log_tmp is not empty
                    log_tmp.pop()  # remove the last \n
                    line = 'The following people have not be recognized from ' + str((self.num_frame - time_slot)//int(fps)) + \
                        's to ' + str(self.num_frame//int(fps)) + 's:\n' + "".join(log_tmp)
                    self.logQueue.put(line)
                    fs.write(line + '\n')
                else:
                    line = 'all students can be recognized from ' + str((self.num_frame - time_slot)//int(fps)) + \
                            's to ' + str(self.num_frame//int(fps)) + 's'
                    self.logQueue.put(line)
                    fs.write(line + '\n')
                fs.close()

            self.num_frame += 1
            cv2.waitKey(1)
        self.logQueue.put('Success: faceRecognition')
        self.isFinishTest = True

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


    # display image
    def displayImage(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # opencv use BGR as default color and change to RGB
        # default: The image is stored using 8-bit indexes into a colormap, for example: a gray image
        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:  # shape[0]:height, shape[1]:width, shape[2]:channels
            if img.shape[2] == 4:  # with alpha channel
                qformat = QImage.Format_RGBA8888  # stored using a 32-bit byte-ordered RGBA format (8-8-8-8)
            else:
                qformat = QImage.Format_RGB888

        # strides[0]: byte per line, strides[1]: byte per pixel, strides[2]: byte per channel
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        self.faceDetectCaptureLabel.setPixmap(QPixmap.fromImage(outImage))
        self.faceDetectCaptureLabel.setScaledContents(True)

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



if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = core()
    window.show()
    sys.exit(app.exec())



