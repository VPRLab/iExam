# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'format.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_format(object):
    def setupUi(self, Core):
        Core.setObjectName("Core")
        Core.resize(512, 248)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        font.setKerning(True)
        Core.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Core.setWindowIcon(icon)
        self.operationGroupBox = QtWidgets.QGroupBox(Core)
        self.operationGroupBox.setGeometry(QtCore.QRect(60, 20, 401, 211))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.operationGroupBox.setFont(font)
        self.operationGroupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.operationGroupBox.setObjectName("operationGroupBox")
        self.layoutWidget = QtWidgets.QWidget(self.operationGroupBox)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 40, 341, 91))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.rowGroupBox = QtWidgets.QGroupBox(self.layoutWidget)
        self.rowGroupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.rowGroupBox.setObjectName("rowGroupBox")
        self.rowLineEdit = QtWidgets.QLineEdit(self.rowGroupBox)
        self.rowLineEdit.setGeometry(QtCore.QRect(0, 40, 161, 31))
        self.rowLineEdit.setObjectName("rowLineEdit")
        self.horizontalLayout.addWidget(self.rowGroupBox)
        self.columnGroupBox = QtWidgets.QGroupBox(self.layoutWidget)
        self.columnGroupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.columnGroupBox.setObjectName("columnGroupBox")
        self.columnLineEdit = QtWidgets.QLineEdit(self.columnGroupBox)
        self.columnLineEdit.setGeometry(QtCore.QRect(0, 40, 161, 31))
        self.columnLineEdit.setObjectName("columnLineEdit")
        self.horizontalLayout.addWidget(self.columnGroupBox)
        self.okButton = QtWidgets.QPushButton(self.operationGroupBox)
        self.okButton.setGeometry(QtCore.QRect(120, 140, 161, 31))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.okButton.setFont(font)
        self.okButton.setMouseTracking(False)
        self.okButton.setObjectName("okButton")
        self.msgLabel = QtWidgets.QLabel(self.operationGroupBox)
        self.msgLabel.setGeometry(QtCore.QRect(40, 180, 331, 20))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        font.setKerning(True)
        self.msgLabel.setFont(font)
        self.msgLabel.setText("")
        self.msgLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.msgLabel.setObjectName("msgLabel")

        self.retranslateUi(Core)
        QtCore.QMetaObject.connectSlotsByName(Core)

    def retranslateUi(self, Core):
        _translate = QtCore.QCoreApplication.translate
        Core.setWindowTitle(_translate("Core", "iExam"))
        self.operationGroupBox.setTitle(_translate("Core", "Input video format"))
        self.rowGroupBox.setTitle(_translate("Core", "Row"))
        self.rowLineEdit.setText(_translate("Core", "5"))
        self.columnGroupBox.setTitle(_translate("Core", "Column"))
        self.columnLineEdit.setText(_translate("Core", "5"))
        self.okButton.setText(_translate("Core", "Submit"))

