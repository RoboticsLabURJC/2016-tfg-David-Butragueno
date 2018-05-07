
# -*- coding: utf-8 -*-

from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import numpy
import sys

class Gui(QtGui.QWidget):

    updGUI=QtCore.pyqtSignal()

    def __init__(self, camera, parent=None):

        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle("Detection Component")
        self.resize(1200,500)
        self.move(150,50)
        self.updGUI.connect(self.update)

        # MAIN IMAGE
        self.imgPrincipal = QtGui.QLabel(self)
        self.imgPrincipal.resize(450,350)
        self.imgPrincipal.move(25,90)
        self.imgPrincipal.show()

        # DETECTION IMAGE
        self.imgDetection = QtGui.QLabel(self)
        self.imgDetection.resize(450,350)
        self.imgDetection.move(725,90)
        self.imgDetection.show()

        # BUTTON
        self.buttonDetection = QtGui.QPushButton('Continuos', self)
        self.buttonDetection.clicked.connect(self.toggle)
        self.buttonDetection.move(550,100)
        self.buttonDetection.setStyleSheet('QPushButton {color:red;}')


        # Configuracion BOX
        #vbox = QtGui.QVBoxLayout()
        #vbox.addWidget(self.imgPrincipal)
        #vbox.addWidget(self.imgDetection)
        #vbox.addWidget(self.button)
        #self.setLayout(vbox)

        self.image_detec = numpy.zeros((1000, 600), dtype=numpy.uint8) 

        self.camera = camera

    def setCamera(self,camera, t_camera):

        self.camera=camera
        self.t_camera=t_camera

    def setDetector(self,detector,t_detector):

        self.detector=detector
        self.t_detector = t_detector

    def update(self): #This function update the GUI for every time the thread change

        image = self.camera.getImage()
        img_out = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)

        scaledImageOut = img_out.scaled(self.imgPrincipal.size())
        self.imgPrincipal.setPixmap(QtGui.QPixmap.fromImage(scaledImageOut))


        image_detec = QtGui.QImage(self.image_detec.data, self.image_detec.shape[1], self.image_detec.shape[0], QtGui.QImage.Format_RGB888)

        scaledImageOut_Detection = image_detec.scaled(self.imgDetection.size())
        self.imgDetection.setPixmap(QtGui.QPixmap.fromImage(scaledImageOut_Detection))


    def toggle(self):

    	self.t_detector.handleButtonDetection()

        if self.t_detector.is_activated:
            self.buttonDetection.setStyleSheet('QPushButton {color:green;}')
        else:
            self.buttonDetection.setStyleSheet('QPushButton {color:red;}')

