
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
        self.resize(1000,600)
        self.move(150,50)
        self.updGUI.connect(self.update)


        # BUTTON
        self.button = QtGui.QPushButton('Pulsa para deteccion', self)
        self.button.clicked.connect(self.handleButton)

        #Original Image Label
        #self.imgLabel=QtGui.QLabel(self)
        #self.imgLabel.resize(640,480)
        #self.imgLabel.move(150,50)
        #self.imgLabel.show()

        # IMAGE PRINCIPAL
        self.imgPrincipal = QtGui.QLabel(self)
        self.imgPrincipal.resize(640,480)
        self.imgPrincipal.move(150,50)
        self.imgPrincipal.show()

        # Configuracion BOX
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.imgPrincipal)
        vbox.addWidget(self.button)
        self.setLayout(vbox) 


    def setCamera(self,camera):
        self.camera=camera

    def update(self): #This function update the GUI for every time the thread change
        image = self.camera.getImage()
        img_out = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
        scaledImageOut = img_out.scaled(self.imgPrincipal.size())
        self.imgPrincipal.setPixmap(QtGui.QPixmap.fromImage(scaledImageOut))

    def handleButton(self):
    	self.camera.handleButton()
