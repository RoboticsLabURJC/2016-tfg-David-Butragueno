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

import json
from pprint import pprint

import re

imageFile = '/home/davidbutra/cocostuff/dataset/images/COCO_train2014_000000000821.jpg'
jsonFile = json.load(open('/home/davidbutra/cocostuff/dataset/annotations-json/cocostuff-10k-v1.1.json'))
pathImages = '/home/davidbutra/cocostuff/dataset/images/'

person_jaccard_index = 0
nperson = 0


class DetectionNet:

    def __init__(self):

        labelmap_file = '/home/davidbutra/caffe/data/VOC0712/labelmap_voc.prototxt'
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

        model_def = '/home/davidbutra/caffe/models/VGGNet/VOC0712/SSD_300x300_coco/deploy.prototxt'
        model_weights = '/home/davidbutra/caffe/models/VGGNet/VOC0712/SSD_300x300_coco/VGG_coco_SSD_300x300.caffemodel'

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


    def detectiontest(self,img, pathImage):

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
        top_label_indices = det_label[top_indices].tolist()
        top_labels = self.get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        print "LABELS DETECTION"
        print(top_labels)

        colors = plt.cm.hsv(np.linspace(0, 1, 81)).tolist()
        font = cv2.FONT_HERSHEY_SIMPLEX

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

        print "GROUND DETECTION"
        print ground_detection

        #Get image id
        image_id = self.get_image_id(pathImage)
        print "IMAGE_ID"
        print image_id

        #Recover image annotations
        image_annotations = self.annotations(image_id)
        print "IMAGE ANNOTATIONS"
        print image_annotations

        #Save labels and id
        jsonFileCategories = jsonFile["categories"]
        labels_ids_array = []
        labels_ids_array_total = []
        labels_array = []
        for c in jsonFileCategories:
            labels_ids_array.extend([c["name"], c["id"]])
            labels_ids_array_total.append(labels_ids_array)
            labels_ids_array = []
            labels_array.append(c["name"])


        #Recover id
        id_array = self.recover_id(top_labels, labels_ids_array_total)
        print "IDS DETECTADOS"
        print id_array

        #GROUND TRUTH
        ground_truth = self.get_ground_truth(id_array, ground_detection, image_id, image_annotations)
        print "GROUND TRUTH"
        print ground_truth

        print "GROUND DETECTION"
        print ground_detection

        truth_area = self.area(ground_truth)
        print "TRUTH AREA"
        print truth_area

        detection_area = self.area(ground_detection)
        print "DETECTED AREA"
        print detection_area

        area_intersection_array = self.area_intersection(ground_truth, ground_detection)
        print "AREA INTERSECTION"
        print area_intersection_array

        area_total_array = self.area_total(truth_area, detection_area, area_intersection_array)
        print "AREA TOTAL"
        print area_total_array

        jaccard_index = self.jaccard_index(area_total_array, area_intersection_array)
        print "JACCARD INDEX"
        print jaccard_index

        return jaccard_index, top_labels


    def jaccard_index(self, area_total_array, area_intersection_array):

        jaccard_index_final = []
        
        for i in range(len(area_total_array)):

            jaccard_index = area_intersection_array[i] / area_total_array[i]
            jaccard_index_final. append(jaccard_index)

        return jaccard_index_final

    def area_total(self, real_area, detected_area, area_intersection):

        area_total_final = []

        for i in range(len(real_area)):

            area_total = real_area[i] + detected_area[i] - area_intersection[i]
            area_total_final.append(area_total)

        return area_total_final

    def area_intersection(self, ground_truth, ground_detection):

        area_intersection = []

        for i in range(len(ground_detection)):

            if ground_truth[i][0] < ground_detection[i][0]:
                xmin = ground_truth[i][0]
            else:
                xmin = ground_detection[i][0]

            if ground_truth[i][1] > ground_detection[i][1]:
                ymin = ground_truth[i][1]
            else:
                ymin = ground_detection[i][1]

            if ground_truth[i][2] < ground_detection[i][2]:
                xmax = ground_truth[i][2]
            else:
                xmax = ground_detection[i][2]

            if ground_truth[i][3] < ground_detection[i][3]:
                ymax = ground_truth[i][3]
            else:
                ymax = ground_detection[i][3]

            area = (xmax - xmin) * (ymax - ymin)

            area_intersection.append(area)

        return area_intersection


    def area(self, array_positions):

        area_array = []

        for i in range(len(array_positions)):

            weight = float(array_positions[i][2]) - float(array_positions[i][0])
            height = float(array_positions[i][3]) - float(array_positions[i][1])
            area = weight * height
            
            area_array.append(area)

        return area_array

    def get_image_id(self, image_file):

        image_file = re.split('.jpg', image_file)
        image_file = re.split('_', image_file[0])
        image_id = int(image_file[2])

        return image_id

    def annotations(self, image_id):

        annotations = []
        annotations_final = []

        jsonFileAnnotations = jsonFile["annotations"]

        for c in jsonFileAnnotations:
            if c["image_id"] == image_id:
                if len(c['bbox']) == 1:
                    #print c['bbox']
                    #print c['bbox'][0]
                    #print c['bbox'][1]
                    #print c['bbox'][2]
                    #print c['bbox'][3]
                    annotations.append(c['category_id'])
                    annotations.append(c['bbox'][0][0])
                    annotations.append(c['bbox'][0][1])
                    annotations.append(c['bbox'][0][2])
                    annotations.append(c['bbox'][0][3])
                    annotations.append(c['area'])
                    annotations_final.append(annotations)
                    annotations = []
                else:
                    print c
                    #print c['bbox'][0]
                    #print c['bbox'][1]
                    #print c['bbox'][2]
                    #print c['bbox'][3]
                    annotations.append(c['category_id'])
                    annotations.append(c['bbox'][0])
                    annotations.append(c['bbox'][1])
                    annotations.append(c['bbox'][2])
                    annotations.append(c['bbox'][3])
                    annotations.append(c['area'])
                    annotations_final.append(annotations)
                    annotations = []

        return annotations_final

    def get_ground_truth(self, id_array, ground_detection, image_id, image_annotations):
        
        ground_truth = []
        ground_truth_final = []
        image_annotations_complete = image_annotations

        for i in range(len(id_array)):
            for c in range(len(image_annotations)):
                print "POSITION ARRAY ANNOTATIONS"
                print c
                print "DETECTED ANNOTATION ID"
                print id_array[i]
                print "ANNOTATION ID"
                print image_annotations[c][0]
                if id_array[i] == image_annotations[c][0]:
                    xmin = int(image_annotations[c][1])
                    ymin = int(image_annotations[c][2])
                    xmax = xmin + int(image_annotations[c][3])
                    ymax = ymin + int(image_annotations[c][4])
                    print "ANNOTATIONS POSITIONS"
                    print str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax)
                    print ground_detection[i]
                    if abs(xmin - ground_detection[i][0]) <= 300 and abs(ymin - ground_detection[i][1]) <= 300 and abs(xmax - ground_detection[i][2]) <= 300 and abs(ymax - ground_detection[i][3]) <= 300:
                        ground_truth.append(xmin)
                        ground_truth.append(ymin)
                        ground_truth.append(xmax)
                        ground_truth.append(ymax)
                        ground_truth_final.append(ground_truth)
                        ground_truth = []
                        print "OK"
                        image_annotations = image_annotations_complete
                        break

        return ground_truth_final

    def recover_id(self, labels_detected_array, labels_ids_array):
        id_array = []
        for c in range(len(labels_detected_array)):
            for i in range(len(labels_ids_array)):
                print labels_ids_array[i]
                if labels_detected_array[c] == labels_ids_array[i][0] :
                    id_array.append(labels_ids_array[i][1])
                    break

        return id_array


net = DetectionNet()

dirsImages = sorted(os.listdir(pathImages))
print dirsImages
detec_array = []

print len(dirsImages)
for x in range(len(dirsImages)):
    if x == 10:
        break
    print x
    print "IMAGEN DE ENTRADA"
    print pathImages + dirsImages[x]
    image = cv2.imread(pathImages + dirsImages[x])
    #img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #img_rgb = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)
    detec = net.detectiontest(image, pathImages + dirsImages[x])
    detec_array.append(detec)
    print detec

jaccardIndex_file = open('jaccardIndexCOCO.txt', 'w')
jaccardIndex_file.write("Jaccard Index COCO\n")
for x in range(len(detec_array)):
    for i in range(len(detec_array[x][0])):
        jaccardIndex_file.write("%s" % str(detec_array[x][1][i]) + ": " + "%s\n" % str(detec_array[x][0][i]))
        if detec_array[x][1][i] == "person":
            person_jaccard_index = person_jaccard_index + detec_array[x][0][i]
            nperson = nperson + 1


jaccardIndex_file.close()

jaccardIndex_average_file = open('jaccardIndex_average_COCO.txt', 'w')

jaccardIndex_average_file.write("Jaccard Index Average COCO\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("PERSON\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(nperson) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(person_jaccard_index/nperson) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.close()
