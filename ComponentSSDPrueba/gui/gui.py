
# -*- coding: utf-8 -*-

from PyQt4 import QtGui
from PyQt4 import QtCore
import numpy
import sys

class Gui(QtGui.QWidget):

    updGUI=QtCore.pyqtSignal()

    def __init__(self, parent=None):

        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle("Detection")
        self.resize(1210,740)
        self.move(0,0)
        self.updGUI.connect(self.update)


        # BUTTON
        self.button = QtGui.QPushButton('Pulsa para deteccion', self)
        self.button.clicked.connect(self.handleButton)

        # BUTTON
        self.buttonMemory = QtGui.QPushButton('Deteccion continua', self)
        self.buttonMemory.clicked.connect(self.handleButtonMemory)

        #Original Image Label
        #self.imgLabel=QtGui.QLabel(self)
        #self.imgLabel.resize(640,480)
        #self.imgLabel.move(150,50)
        #self.imgLabel.show()

        # IMAGE PRINCIPAL
        self.imgPrincipal = QtGui.QLabel(self)
        self.imgPrincipal.resize(1210,370)
        self.imgPrincipal.move(0,0)
        self.imgPrincipal.show()

        # IMAGE DETECTION
        self.imgDetection = QtGui.QLabel(self)
        self.imgDetection.resize(1210,370)
        self.imgDetection.move(0,370)
        self.imgDetection.show()

        # Configuracion BOX
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.imgPrincipal)
        vbox.addWidget(self.imgDetection)
        vbox.addWidget(self.button)
        vbox.addWidget(self.buttonMemory)
        self.setLayout(vbox) 


    def setCamera(self,camera):
        self.camera=camera

    def setDetector(self,detector):
        self.detector=detector

    def update(self): #This function update the GUI for every time the thread change
        image = self.camera.getImage()
        img_out = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)

        scaledImageOut = img_out.scaled(self.imgPrincipal.size())
        self.imgPrincipal.setPixmap(QtGui.QPixmap.fromImage(scaledImageOut))

        image_detec = self.detector.getImageDetected()
        img_detec_out = QtGui.QImage(image_detec.data, image_detec.shape[1], image_detec.shape[0], QtGui.QImage.Format_RGB888)

        scaledImageOut_Detection = img_detec_out.scaled(self.imgDetection.size())
        self.imgDetection.setPixmap(QtGui.QPixmap.fromImage(scaledImageOut_Detection))

    def handleButton(self):
    	self.camera.handleButton()

    def handleButtonMemory(self):
    	self.detector.handleButtonMemory()
