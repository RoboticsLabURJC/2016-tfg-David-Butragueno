
# -*- coding: utf-8 -*-

from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4 import Qt
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from PyQt4.Qt import *
import numpy
import sys
import cv2

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

        #FPS MAIN IMAGE
        self.fpsImgPrincipal = QtGui.QLabel(self)
        self.fpsImgPrincipal.resize(50,40)
        self.fpsImgPrincipal.move(220,450)
        self.fpsImgPrincipal.show()

        # DETECTED IMAGE
        self.imgDetection = QtGui.QLabel(self)
        self.imgDetection.resize(450,350)
        self.imgDetection.move(725,90)
        self.imgDetection.show()

        #FPS DETECTED IMAGE
        self.fpsImgDetection = QtGui.QLabel(self)
        self.fpsImgDetection.resize(50,40)
        self.fpsImgDetection.move(930,450)
        self.fpsImgDetection.show()

        # CONTINUOUS DETECTION BUTTON
        self.buttonDetection = QtGui.QPushButton(self)
        self.buttonDetection.setText('Continuos')
        self.buttonDetection.clicked.connect(self.toggle)
        self.buttonDetection.move(550,100)
        self.buttonDetection.setStyleSheet('QPushButton {color:red;}')

        # ONE DETECTION BUTTON
        self.buttonDetection = QtGui.QPushButton(self)
        self.buttonDetection.setText('Detect Once')
        self.buttonDetection.clicked.connect(self.detectOnce)
        self.buttonDetection.move(550,410)
        #self.buttonDetection.setStyleSheet('QPushButton {color:red;}')

        #JDEROBOT IMAGE
        self.logo_label = QtGui.QLabel(self)
	self.logo_label.resize(150, 150)
        self.logo_label.move(520, 200)
	self.logo_label.setScaledContents(True)
	
	logo_img = QtGui.QImage()
	logo_img.load('/home/davidbutra/Escritorio/JdeRobot.png')
	self.logo_label.setPixmap(QtGui.QPixmap.fromImage(logo_img))
	self.logo_label.show()


        # Configuracion BOX
        #vbox = QtGui.QVBoxLayout()
        #vbox.addWidget(self.imgPrincipal)
        #vbox.addWidget(self.imgDetection)
        #vbox.addWidget(self.button)
        #self.setLayout(vbox)

        #self.image_detec = numpy.zeros((1000, 600), dtype=numpy.uint8) 

        self.camera = camera

    def setCamera(self,camera, t_camera):

        self.camera=camera
        self.t_camera=t_camera

    def setDetector(self,detector,t_detector):

        self.detector=detector
        self.t_detector = t_detector

    def update(self): #This function update the GUI for every time the thread change

        self.image = self.camera.getImage()
        img_out = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888)

        scaledImageOut = img_out.scaled(self.imgPrincipal.size())
        self.imgPrincipal.setPixmap(QtGui.QPixmap.fromImage(scaledImageOut))

        if self.t_detector.is_activated:
            self.image_detec = self.detector.detectiontest(self.detector.img)
            img_detected = QtGui.QImage(self.image_detec.data, self.image_detec.shape[1], self.image_detec.shape[0], QtGui.QImage.Format_RGB888)
            image_detec_final = img_detected.scaled(self.imgDetection.size())

            self.imgDetection.setPixmap(QtGui.QPixmap.fromImage(image_detec_final))

        elif not self.t_detector.runOnce_activated:

            image_detec_final = QtGui.QImage()
            image_detec_final.load('/home/davidbutra/Escritorio/JdeRobot.png')
            image_detec_final = image_detec_final.scaled(self.imgDetection.size())

            self.imgDetection.setPixmap(QtGui.QPixmap.fromImage(image_detec_final))

        #self.fpsImgPrincipal.setText("%d FPS" % (self.t_camera.framerate))
        #self.fpsImgDetection.setText("%d FPS" % (self.t_detector.framerate))


    def toggle(self):

    	self.t_detector.handleButtonDetection()
        self.t_detector.runOnce_activated = False

        if self.t_detector.is_activated:
            self.buttonDetection.setStyleSheet('QPushButton {color:green;}')
        else:
            self.buttonDetection.setStyleSheet('QPushButton {color:red;}')


    def detectOnce(self):

        if not self.t_detector.is_activated:

            self.t_detector.runOnce_activated = True

            self.image_detec = self.detector.detectiontest(self.detector.img)
            img_detected = QtGui.QImage(self.image_detec.data, self.image_detec.shape[1], self.image_detec.shape[0], QtGui.QImage.Format_RGB888)
            image_detec_final = img_detected.scaled(self.imgDetection.size())

            self.imgDetection.setPixmap(QtGui.QPixmap.fromImage(image_detec_final))

 

