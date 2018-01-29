import sys, traceback, Ice

import time
import matplotlib.pyplot as plt

import jderobot
import numpy as np
import threading
import cv2
import Image
sys.path.insert(0, '/home/davidbutra/caffe/python')
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2

import cv2

input_image = '/home/davidbutra/data/VOCdevkit/VOC2012/JPEGImages/2007_009938.jpg'
input_imageII = '/home/davidbutra/data/VOCdevkit/VOC2012/JPEGImages/2007_000187.jpg'

class Detector():

    def __init__(self):
        self.handleButtonON = False

    def getImageDetected(self): 

        if True:
            if self.handleButtonON:
                image = cv2.imread(input_image)
            else:
                image = cv2.imread(input_imageII)

        return image
        

    def update(self): #Updates the camera every time the thread changes
        
        if self.detector:
            self.lock.acquire()
            self.image = self.detector.getImageData("RGB8")
            self.height= self.image.description.height
            self.width = self.image.description.width
            self.lock.release()

    def handleButtonMemory(self):
        self.handleButtonON = not self.handleButtonON
