import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/davidbutra/caffe/python')
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2

import cv2
import xml.etree.ElementTree as ET

input_image = '/home/davidbutra/data/VOCdevkit/VOC2012/JPEGImages/2007_000783.jpg'
pathImages = '/home/davidbutra/data/VOCdevkit/VOC2012/JPEGImages/'
pathAnnotations = '/home/davidbutra/data/VOCdevkit/VOC2012/Annotations/'

class DetectionNet:

    def __init__(self):

        labelmap_file = '/home/davidbutra/caffe/data/VOC0712/labelmap_voc.prototxt'
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

        model_def = '/home/davidbutra/caffe/models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
        model_weights = '/home/davidbutra/caffe/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'

        self.net = caffe.Net(model_def,      # defines the structure of the model
                        model_weights,  # contains the trained weights
                        caffe.TEST)     # use test mode (e.g., don't perform dropout)

        # fin caffe


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


    def detectiontest(self,img,annt):


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

        #print ('Executed in',  str((endA - startA)*1000))


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

        ground_detection = []
        #for i in xrange(top_conf.shape[0]):
        for i in range(top_conf.shape[0]):
            positions = []
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            positions.append(xmin)
            positions.append(ymin)
            positions.append(xmax)
            positions.append(ymax)
            ground_detection.append(positions)

            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),2) #Color BGR

        print "OBJETOS DETECTADOS"
        print top_labels

        ground_truth = self.get_ground_truth(input_image, annt, top_labels)

        print "GROUND_TRUTH"
        print ground_truth
        print "GROUND_DETECTION"
        print ground_detection

        rectangle = self.rectangle_intersection(ground_truth, ground_detection)

        print rectangle

        area_intersection = self.area(rectangle)

        print area_intersection

        area_total = self.area_total(ground_truth, ground_detection, area_intersection)

        print area_total

        jaccard_index = self.jaccard_index(area_total, area_intersection)

        print jaccard_index


        return jaccard_index, ground_truth[1]


    def jaccard_index(self, area_total, area_intersection):

        jaccard_index_array = []

        for x in range(len(area_total)):

            jaccard_index = float(area_intersection[x]) / float(area_total[x])

            jaccard_index_array.append(jaccard_index)

        return jaccard_index_array


    def area_total(self, rectangle1, rectangle2, area_intersection):

        area_total_array = []

        for x in range(len(rectangle1[0])):

            heigh_rectangle1 = int(rectangle1[0][x][3]) - int(rectangle1[0][x][1]) 
            weigh_rectangle1 = int(rectangle1[0][x][2]) - int(rectangle1[0][x][0])

            area_rectangle1 = heigh_rectangle1 * weigh_rectangle1

            heigh_rectangle2 = int(rectangle2[x][3]) - int(rectangle2[x][1]) 
            weigh_rectangle2 = int(rectangle2[x][2]) - int(rectangle2[x][0])

            area_rectangle2 = heigh_rectangle2 * weigh_rectangle2

            area_total = area_rectangle1 + area_rectangle2 - area_intersection[x]

            area_total_array.append(area_total)

        return area_total_array
        

    def area(self, rectangles_array):

        area_array = []

        for x in range(len(rectangles_array)):

            heigh = int(rectangles_array[x][3]) - int(rectangles_array[x][1]) 
            weigh = int(rectangles_array[x][2]) - int(rectangles_array[x][0])

            area = heigh * weigh

            area_array.append(area)

        return area_array

    def rectangle_intersection(self, rectangles_array1, rectangles_array2):

        positions_array = []

        for x in range(len(rectangles_array1[0])):

            if int(rectangles_array1[0][x][0]) >= int(rectangles_array2[x][0]):
                xmin_intersection = int(rectangles_array1[0][x][0])
            else:
                xmin_intersection = int(rectangles_array2[x][0])

            if int(rectangles_array1[0][x][1]) >= int(rectangles_array2[x][1]):
                ymin_intersection = int(rectangles_array1[0][x][1])
            else:
                ymin_intersection = int(rectangles_array2[x][1])

            if int(rectangles_array1[0][x][2]) <= int(rectangles_array2[x][2]):
                xmax_intersection = int(rectangles_array1[0][x][2])
            else:
                xmax_intersection = int(rectangles_array2[x][2])

            if int(rectangles_array1[0][x][3]) <= int(rectangles_array2[x][3]):
                ymax_intersection = int(rectangles_array1[0][x][3])
            else:
                ymax_intersection = int(rectangles_array2[x][3])

            positions = []
            positions.append(xmin_intersection)
            positions.append(ymin_intersection)
            positions.append(xmax_intersection)
            positions.append(ymax_intersection)
            positions_array.append(positions)

        return positions_array

    def get_ground_truth(self, input_image, annt, label_detection):

        file = open(annt, 'r')

        tree = ET.parse(file)
        root = tree.getroot()

        positions_array = []
        names_array = []

        for x in range(len(label_detection)):
            print "Buscando etiqueta detectada: " + label_detection[x]
            for element in root.findall('object'):
                name = element.find('name').text
                print "Elemento del XML: " + name
                if name == label_detection[x]:
                    print "Encontrado: " + label_detection[x] + " = " + name 
                    xmin = element[4][0].text
                    ymin = element[4][1].text
                    xmax = element[4][2].text
                    ymax = element[4][3].text
                    print xmin, ymin, xmax, ymax
                    positions = []
                    positions.append(xmin)
                    positions.append(ymin)
                    positions.append(xmax)
                    positions.append(ymax)
                    positions_array.append(positions)
                    names_array.append(name)
                    break
            #break
        return positions_array, names_array


net = DetectionNet()

dirsImages = sorted(os.listdir(pathImages))
dirsAnnotations = sorted(os.listdir(pathAnnotations))
detec_array = []

for x in range(len(dirsImages)):
    if x == 5:
        break
    image = cv2.imread(pathImages + dirsImages[x])
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)
    detec = net.detectiontest(image, pathAnnotations + dirsAnnotations[x])
    detec_array.append(detec)
    print detec

jaccardIndex_file = open('jaccardIndex.txt', 'w')
jaccardIndex_file.write("Jaccard Index\n")
for x in range(len(detec_array)):
    for i in range(len(detec_array[x][0])):
        jaccardIndex_file.write("%s" % str(detec_array[x][1][i]) + ": " + "%s\n" % str(detec_array[x][0][i]))
