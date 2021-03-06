# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'faceRecognition.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_faceRfecognition(object):
    def setupUi(self, Core):
        Core.setObjectName("Core")
        Core.resize(1280, 656)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        font.setKerning(True)
        Core.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Core.setWindowIcon(icon)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Core)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.videoGroupBox = QtWidgets.QGroupBox(Core)
        self.videoGroupBox.setEnabled(True)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.videoGroupBox.setFont(font)
        self.videoGroupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.videoGroupBox.setFlat(False)
        self.videoGroupBox.setObjectName("videoGroupBox")
        self.faceDetectCaptureLabel = QtWidgets.QLabel(self.videoGroupBox)
        self.faceDetectCaptureLabel.setGeometry(QtCore.QRect(0, 30, 960, 591))
        self.faceDetectCaptureLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.faceDetectCaptureLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.faceDetectCaptureLabel.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse)
        self.faceDetectCaptureLabel.setObjectName("faceDetectCaptureLabel")
        self.operationGroupBox = QtWidgets.QGroupBox(self.videoGroupBox)
        self.operationGroupBox.setGeometry(QtCore.QRect(970, 20, 291, 601))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.operationGroupBox.setFont(font)
        self.operationGroupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.operationGroupBox.setObjectName("operationGroupBox")
        self.logGroupBox = QtWidgets.QGroupBox(self.operationGroupBox)
        self.logGroupBox.setGeometry(QtCore.QRect(0, 290, 281, 311))
        self.logGroupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.logGroupBox.setObjectName("logGroupBox")
        self.logTextEdit = QtWidgets.QTextEdit(self.logGroupBox)
        self.logTextEdit.setGeometry(QtCore.QRect(0, 30, 281, 271))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        font.setKerning(True)
        self.logTextEdit.setFont(font)
        self.logTextEdit.setFrameShape(QtWidgets.QFrame.Box)
        self.logTextEdit.setObjectName("logTextEdit")
        self.layoutWidget = QtWidgets.QWidget(self.operationGroupBox)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 30, 271, 243))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.uploadRosterButton = QtWidgets.QPushButton(self.layoutWidget)
        self.uploadRosterButton.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semibold")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.uploadRosterButton.setFont(font)
        self.uploadRosterButton.setCheckable(True)
        self.uploadRosterButton.setObjectName("uploadRosterButton")
        self.verticalLayout.addWidget(self.uploadRosterButton)
        self.clipFormatButton = QtWidgets.QPushButton(self.layoutWidget)
        self.clipFormatButton.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semibold")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.clipFormatButton.setFont(font)
        self.clipFormatButton.setCheckable(True)
        self.clipFormatButton.setObjectName("clipFormatButton")
        self.verticalLayout.addWidget(self.clipFormatButton)
        self.uploadVideoButton = QtWidgets.QPushButton(self.layoutWidget)
        self.uploadVideoButton.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semibold")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.uploadVideoButton.setFont(font)
        self.uploadVideoButton.setCheckable(True)
        self.uploadVideoButton.setObjectName("uploadVideoButton")
        self.verticalLayout.addWidget(self.uploadVideoButton)
        self.faceRecognitionButton = QtWidgets.QPushButton(self.layoutWidget)
        self.faceRecognitionButton.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semibold")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.faceRecognitionButton.setFont(font)
        self.faceRecognitionButton.setCheckable(True)
        self.faceRecognitionButton.setObjectName("faceRecognitionButton")
        self.verticalLayout.addWidget(self.faceRecognitionButton)
        self.drawButton = QtWidgets.QPushButton(self.layoutWidget)
        self.drawButton.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI Semibold")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.drawButton.setFont(font)
        self.drawButton.setCheckable(True)
        self.drawButton.setObjectName("drawButton")
        self.verticalLayout.addWidget(self.drawButton)
        self.horizontalLayout.addWidget(self.videoGroupBox)

        self.retranslateUi(Core)
        QtCore.QMetaObject.connectSlotsByName(Core)

    def retranslateUi(self, Core):
        _translate = QtCore.QCoreApplication.translate
        Core.setWindowTitle(_translate("Core", "iExam"))
        self.videoGroupBox.setTitle(_translate("Core", "Zoom Video"))
        self.faceDetectCaptureLabel.setText(_translate("Core", "<html><head/><body><p><span style=\" color:#ff0000;\">Zoom Video Window</span></p></body></html>"))
        self.operationGroupBox.setTitle(_translate("Core", "Operation"))
        self.logGroupBox.setTitle(_translate("Core", "System Log"))
        self.uploadRosterButton.setText(_translate("Core", "select roster path"))
        self.clipFormatButton.setText(_translate("Core", "input view format"))
        self.uploadVideoButton.setText(_translate("Core", "select training video\n"
"and training dataset"))
        self.faceRecognitionButton.setText(_translate("Core", "select test video and run"))
        self.drawButton.setText(_translate("Core", "draw analysis graph"))

