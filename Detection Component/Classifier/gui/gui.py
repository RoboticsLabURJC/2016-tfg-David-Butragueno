
# -*- coding: utf-8 -*-

from PyQt4 import QtGui
from PyQt4 import QtCore
import numpy
import sys

class Gui(QtGui.QWidget):

    updGUI=QtCore.pyqtSignal()

    def __init__(self, parent=None):

        QtGui.QWidget.__init__(self, parent)
        #self.setScaledContents(True)
        self.setWindowTitle("Detection")
        #self.imgLabel=QtGui.QLabel(self)
        self.resize(1000,600)
        self.move(150,50)
        #self.imgLabel.show()
        self.updGUI.connect(self.update)

        #Original Image Label
        self.imgLabel=QtGui.QLabel(self)
        self.imgLabel.resize(640,480)
        self.imgLabel.move(150,50)
        self.imgLabel.show()


    def setCamera(self,camera):
        self.camera=camera

    def update(self): #This function update the GUI for every time the thread change
        image = self.camera.getImage()
        #image_out = self.camera.detectiontest(image)
        img_out = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
        scaledImageOut = img_out.scaled(self.imgLabel.size())
        self.imgLabel.setPixmap(QtGui.QPixmap.fromImage(scaledImageOut))
