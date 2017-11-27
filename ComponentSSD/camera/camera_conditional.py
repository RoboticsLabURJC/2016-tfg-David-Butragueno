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

labelmap_conditional_location = '/home/davidbutra/caffe/data/VOC0712/labelmap_conditional_camera.prototxt'

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

        labelmap_file = '/home/davidbutra/caffe/data/VOC0712/labelmap_voc.prototxt'
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

        array_labels = self.get_arraylabels_conditional(self.labelmap, Labels)
        print array_labels

        labelmap = self.get_labelmap_conditional(self.labelmap, array_labels)

        labelmapfile=open(labelmap_conditional_location, 'w')

        for x in range(0, len(labelmap[0])):
                labelmapfile.write("item {\r\n")
                labelmapfile.write('  name: ' + '"' + labelmap[0][x] + '"' + '\r\n')
                labelmapfile.write("  label: " + str(labelmap[1][x]) + "\n")
                labelmapfile.write('  display_name: ' + '"' + labelmap[2][x] + '"' + '\r\n')
                labelmapfile.write("}\r\n")
    
        labelmapfile.close()

        file = open(labelmap_conditional_location, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

        #Net parameters necesary
        model_def = '/home/davidbutra/caffe/models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
        model_weights = '/home/davidbutra/caffe/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'

        self.net = caffe.Net(model_def,      # defines the structure of the model
                        model_weights,  # contains the trained weights
                        caffe.TEST)     # use test mode (e.g., don't perform dropout)

        #self.height= self.camera.getHeight()
        #self.width = self.camera.getWidth()
        #self.image = 0

        #if self.camera.hasproxy():
         #   self.trackImage = np.zeros((self.height, self.width,3), np.uint8)
         #   self.trackImage.shape = self.height, self.width, 3

          #  self.thresoldImage = np.zeros((self.height,self. width,1), np.uint8)
           # self.thresoldImage.shape = self.height, self.width,

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
            #assert found == True
        return labelnames

    def get_arraylabels_conditional(self, labelmap, array_labels):

        array_labels_YesNo = []
        array_labels = array_labels.split(" ")

        for x in range(0, len(labelmap.item)):
            found = False
            for i in range(0, len(array_labels)):
                if labelmap.item[x].display_name == array_labels[i]:
                    array_labels_YesNo.append(1)
                    found = True
                    break
            if found == False:
                array_labels_YesNo.append(0)

        return array_labels_YesNo

    # Funcion obtener los nombres y etiquetas de los objetos que queremos detectar
    def get_labelmap_conditional(self,labelmap, array_conditional):

        names = []
        labels = []
        display = []
        numlabels = len(labelmap.item)

        for i in xrange(0, numlabels):
            if array_conditional[i] == 1:
                names.append(labelmap.item[i].name)
                labels.append(labelmap.item[i].label)
                display.append(labelmap.item[i].display_name)

        return names, labels, display

    def delete_labels(self,labelmap, labels, top_indices):

        labels_conditional = []
        top_indices_conditional = []
        for i in range(0, len(labels)):
            for x in range(0, len(labelmap.item)):
                if labels[i] == labelmap.item[x].label:
                    labels_conditional.append(labelmap.item[x].label)
                    top_indices_conditional.append(top_indices[i])
                    break

        return labels_conditional, top_indices_conditional


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
        detections = self.net.forward()['detection_out']


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
        
        print ("Confianza y top indices original")
        print top_conf
        print top_indices

        print ("Etiquetas detectadas")
        top_label_indices = det_label[top_indices].tolist()
        print top_label_indices

        top_label_indices_conditional = self.delete_labels(self.labelmap, top_label_indices, top_indices)
        print ("Etiquetas que queremos detectar y top indices")
        print top_label_indices_conditional

        top_labels = self.get_labelname(self.labelmap, top_label_indices)
        print ("Nombre de las etiquetas seleccionadas")
        print top_labels

        top_xmin = det_xmin[top_label_indices_conditional[1]]
        top_ymin = det_ymin[top_label_indices_conditional[1]]
        top_xmax = det_xmax[top_label_indices_conditional[1]]
        top_ymax = det_ymax[top_label_indices_conditional[1]]


        colors = plt.cm.hsv(np.linspace(0, 1, 81)).tolist()
        font = cv2.FONT_HERSHEY_SIMPLEX

        #for i in xrange(top_conf.shape[0]):
        #for i in range(top_conf.shape[0]):
        for i in range(len(top_labels)):
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

        return img


    def getImage(self): #This function gets the image from the webcam and trasformates it for the network
        
        if self.camera:
            self.lock.acquire()
            image = np.zeros((self.height, self.width, 3), np.uint8)
            image = np.frombuffer(self.image.pixelData, dtype=np.uint8)
            image.shape = self.height, self.width, 3
            if self.handleButtonON:
                image = self.detectiontest(image)
            self.lock.release()
        
        '''
        image = self.image
        if self.handleButtonON:
            image = self.detectiontest(image)
        '''
        return image
        

    def update(self): #Updates the camera every time the thread changes
        
        if self.camera:
            self.lock.acquire()
            self.image = self.camera.getImageData("RGB8")
            self.height= self.image.description.height
            self.width = self.image.description.width
            self.lock.release()
        
        '''
        self.lock.acquire()
        self.image = self.camera.getImage()
        self.lock.release()
        '''

    def handleButton(self):
        self.handleButtonON = not self.handleButtonON
