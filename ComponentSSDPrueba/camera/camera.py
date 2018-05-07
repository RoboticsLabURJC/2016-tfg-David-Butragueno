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


class Camera():

    def __init__(self):

        status = 0

        ic = None
        # Initializing the Ice run-time.
        ic = Ice.initialize(sys.argv)
        properties = ic.getProperties()
        print properties
        print (".-------------------------------------------------")
        self.lock = threading.Lock()


        try:
            obj = ic.propertyToProxy("DetectorSSD.Camera.Proxy")
            print obj
            self.camera = jderobot.CameraPrx.checkedCast(obj)
            Labels = properties.getProperty("DetectorSSD.Labels")
            print Labels
            if self.camera:
                self.image = self.camera.getImageData("RGB8")
                self.height= self.image.description.height
                self.width = self.image.description.width
            else:
                print 'Interface camera not connected'

        except:
            traceback.print_exc()
            exit()
            status = 1

        self.handleButtonON = False
        caffe.set_mode_cpu()


    def getImage(self): #This function gets the image from the webcam and trasformates it for the network
        
        if self.camera:
            self.lock.acquire()
            image = np.zeros((self.height, self.width, 3), np.uint8)
            image = np.frombuffer(self.image.pixelData, dtype=np.uint8)
            image.shape = self.height, self.width, 3

        return image
        

    def update(self): #Updates the camera every time the thread changes
        
        if self.camera:
            self.lock.acquire()
            self.image = self.camera.getImageData("RGB8")
            self.height= self.image.description.height
            self.width = self.image.description.width
            self.lock.release()
        

    def handleButton(self):
        self.handleButtonON = not self.handleButtonON
