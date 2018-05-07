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


class Detector():

    def __init__(self, camera, gui):

        self.handleButtonON = False

        ic = None
        # Initializing the Ice run-time.
        ic = Ice.initialize(sys.argv)

        self.gui = gui
        self.camera = camera

        labelmap_file = '/home/davidbutra/caffe/data/VOC0712/labelmap_voc.prototxt'
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

        #Net parameters necesary
        model_def = '/home/davidbutra/caffe/models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
        model_weights = '/home/davidbutra/caffe/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'

        self.net = caffe.Net(model_def,      # defines the structure of the model
                        model_weights,  # contains the trained weights
                        caffe.TEST)     # use test mode (e.g., don't perform dropout)

    def get_labelname(self,labelmap, labels):
        num_labels = len(labelmap.item)
        labelnames = []
        if type(labels) is not list:
            labels = [labels]
        for label in labels:
            found = False
            for i in xrange(0, num_labels):
                if label == labelmap.item[i].label:
                    found = True
                    labelnames.append(labelmap.item[i].display_name)
                    break
            assert found == True
        return labelnames

    def detectiontest(self,img):

        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104,117,123])) # mean pixel


        # set net to batch size of 1
        image_resize = 300
        self.net.blobs['data'].reshape(1,3,image_resize,image_resize)

        transformed_image = transformer.preprocess('data', img)

        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        startA = time.time()
        detections = self.net.forward()['detection_out']
        endA = time.time()

        print ('Executed in',  str((endA - startA)*1000))
        #print detections


        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        top_indices = []
        # Get detections with confidence higher than 0.6.
        for i in range(0, len(det_conf)):
            if (det_conf[i] >= 0.6):
                top_indices.append(i)

        #print(top_indices)

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = self.get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        print(top_labels)

        colors = plt.cm.hsv(np.linspace(0, 1, 81)).tolist()
        font = cv2.FONT_HERSHEY_SIMPLEX

        #for i in xrange(top_conf.shape[0]):
        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            color = colors[label]
            for i in range(0, len(color)-1):
                color[i]=color[i]*255

            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color,2)
            cv2.putText(img,label_name,(xmin+5,ymin+10), font, 0.5,(255,0,0),2)

        #self.handleButtonON = False

        return img
        

    def update(self): #Updates the camera every time the thread changes
        
        img = self.camera.getImage()

        if self.handleButtonON:
            img = self.detectiontest(img)

        self.gui.image_detec = img

    def handleButtonDetection(self):
        self.handleButtonON = not self.handleButtonON
