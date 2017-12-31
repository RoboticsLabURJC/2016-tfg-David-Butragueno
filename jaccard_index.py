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

aeroplane_jaccard_index = 0
naeroplane = 0

bicycle_jaccard_index = 0
nbicycle = 0

bird_jaccard_index = 0
nbird = 0

boat_jaccard_index = 0
nboat = 0

bottle_jaccard_index = 0
nbottle = 0

bus_jaccard_index = 0
nbus = 0

car_jaccard_index = 0
ncar = 0

cat_jaccard_index = 0
ncat = 0

chair_jaccard_index = 0
nchair = 0

cow_jaccard_index = 0
ncow = 0

diningtable_jaccard_index = 0
ndiningtable = 0

dog_jaccard_index = 0
ndog = 0

horse_jaccard_index = 0
nhorse = 0

motorbike_jaccard_index = 0
nmotorbike = 0

person_jaccard_index = 0
nperson = 0

pottedplant_jaccard_index = 0
npottedplant = 0

sheep_jaccard_index = 0
nsheep = 0

sofa_jaccard_index = 0
nsofa = 0

train_jaccard_index = 0
ntrain = 0

tvmonitor_jaccard_index = 0
ntvmonitor = 0

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

        ground_truth = self.get_ground_truth(input_image, annt, top_labels, ground_detection)

        print "GROUND_TRUTH"
        print ground_truth
        print "GROUND_DETECTION"
        print ground_detection

        rectangle = self.rectangle_intersection(ground_truth, ground_detection)

        print "RECTANGLE_INTERSECTION"
        print rectangle

        area_intersection = self.area(rectangle)

        print "AREA_INTERSECTION"
        print area_intersection

        area_total = self.area_total(ground_truth, ground_detection, area_intersection)

        print "AREA_TOTAL"
        print area_total

        jaccard_index = self.jaccard_index(area_total, area_intersection)

        print "JACCARD_INDEX"
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

    def get_ground_truth(self, input_image, annt, label_detection, ground_detection):

        file = open(annt, 'r')

        tree = ET.parse(file)
        root = tree.getroot()

        names_xml = []
        names_xml_final = []

        for element in root.findall('object'):
            name = element.find('name').text
            names_xml.append(name)
            bndbox = element.find('bndbox')
            names_xml.append(bndbox[0].text) #xmin
            names_xml.append(bndbox[1].text) #ymin
            names_xml.append(bndbox[2].text) #xmax
            names_xml.append(bndbox[3].text) #ymax
            #names_xml.append(element[4][0].text) #xmin
            #names_xml.append(element[4][1].text) #ymin
            #names_xml.append(element[4][2].text) #xmax
            #names_xml.append(element[4][3].text) #ymax
            names_xml.append("Not Found")
            names_xml.append("Not Correct Object")
            names_xml_final.append(names_xml)
            names_xml = []

        print "VARIABLE_NOMBRES_XML"
        print names_xml_final


        positions_array = []
        names_array = []

        for x in range(len(label_detection)):
            print "Buscando etiqueta detectada: " + label_detection[x]
            for t in range(len(ground_detection)):
                for i in range(len(names_xml_final)):
                    if abs(int(ground_detection[t][0]) - int(names_xml_final[i][1])) <= 40 and abs(int(ground_detection[t][1]) - int(names_xml_final[i][2])) <= 40 and abs(int(ground_detection[t][2]) - int(names_xml_final[i][3])) <= 40 and abs(int(ground_detection[t][3]) - int(names_xml_final[i][4])) <= 40:
                        names_xml_final[i][6] = "Correct Object"
                        print "OBJETO CORRECTO"
                        break
                    print "Elemento del XML: " + names_xml_final[i][0]
                if names_xml_final[i][0] == label_detection[x] and names_xml_final[i][5] == "Not Found" and names_xml_final[i][6] == "Correct Object":
                    print "Encontrado: " + label_detection[x] + " = " + names_xml_final[i][0] 
                    names_xml_final[i][5] = "Found"
                    names_xml_final[i][6] = "Not Correct Object"
                    name = names_xml_final[i][0]
                    xmin = names_xml_final[i][1]
                    ymin = names_xml_final[i][2]
                    xmax = names_xml_final[i][3]
                    ymax = names_xml_final[i][4]
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
        print "VARIABLE_NOMBRES_XML_FINAL"
        print names_xml_final
        return positions_array, names_array


net = DetectionNet()

dirsImages = sorted(os.listdir(pathImages))
dirsAnnotations = sorted(os.listdir(pathAnnotations))
detec_array = []

for x in range(len(dirsImages)):
    if x == 200:
        break
    print "IMAGEN DE ENTRADA"
    print pathImages + dirsImages[x]
    image = cv2.imread(pathImages + dirsImages[x])
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)
    detec = net.detectiontest(image, pathAnnotations + dirsAnnotations[x])
    detec_array.append(detec)
    print detec

jaccardIndex_file = open('jaccardIndex.txt', 'w')
jaccardIndex_file.write("Jaccard Index\n")
for x in range(len(detec_array)):
    jaccardIndex_file.write(dirsAnnotations[x] +"\n")
    for i in range(len(detec_array[x][0])):
        jaccardIndex_file.write("%s" % str(detec_array[x][1][i]) + ": " + "%s\n" % str(detec_array[x][0][i]))
        if detec_array[x][1][i] == "aeroplane":
            aeroplane_jaccard_index = aeroplane_jaccard_index + detec_array[x][0][i]
            naeroplane = naeroplane + 1

        elif detec_array[x][1][i] == "bicycle":
            bicycle_jaccard_index = bicycle_jaccard_index + detec_array[x][0][i]
            nbicycle = nbicycle + 1

        elif detec_array[x][1][i] == "bird":
            bird_jaccard_index = bird_jaccard_index + detec_array[x][0][i]
            nbird = nbird + 1

        elif detec_array[x][1][i] == "boat":
            boat_jaccard_index = boat_jaccard_index + detec_array[x][0][i]
            nboat = nboat + 1

        elif detec_array[x][1][i] == "bottle":
            bottle_jaccard_index = bottle_jaccard_index + detec_array[x][0][i]
            nbottle = nbottle + 1

        elif detec_array[x][1][i] == "bus":
            bus_jaccard_index = bus_jaccard_index + detec_array[x][0][i]
            nbus = nbus + 1

        elif detec_array[x][1][i] == "car":
            car_jaccard_index = car_jaccard_index + detec_array[x][0][i]
            ncar = ncar + 1

        elif detec_array[x][1][i] == "cat":
            cat_jaccard_index = cat_jaccard_index + detec_array[x][0][i]
            ncat = ncat + 1

        elif detec_array[x][1][i] == "chair":
            chair_jaccard_index = chair_jaccard_index + detec_array[x][0][i]
            nchair = nchair + 1

        elif detec_array[x][1][i] == "cow":
            cow_jaccard_index = cow_jaccard_index + detec_array[x][0][i]
            ncow = ncow + 1

        elif detec_array[x][1][i] == "diningtable":
            diningtable_jaccard_index = diningtable_jaccard_index + detec_array[x][0][i]
            ndiningtable = ndiningtable + 1

        elif detec_array[x][1][i] == "dog":
            dog_jaccard_index = dog_jaccard_index + detec_array[x][0][i]
            ndog = ndog + 1

        elif detec_array[x][1][i] == "horse":
            horse_jaccard_index = horse_jaccard_index + detec_array[x][0][i]
            nhorse = nhorse + 1

        elif detec_array[x][1][i] == "motorbike":
            motorbike_jaccard_index = motorbike_jaccard_index + detec_array[x][0][i]
            nmotorbike = nmotorbike + 1

        elif detec_array[x][1][i] == "person":
            person_jaccard_index = person_jaccard_index + detec_array[x][0][i]
            nperson = nperson + 1

        elif detec_array[x][1][i] == "pottedplant":
            pottedplant_jaccard_index = pottedplant_jaccard_index + detec_array[x][0][i]
            npottedplant = npottedplant + 1

        elif detec_array[x][1][i] == "sheep":
            sheep_jaccard_index = sheep_jaccard_index + detec_array[x][0][i]
            nsheep = nsheep + 1

        elif detec_array[x][1][i] == "sofa":
            sofa_jaccard_index = sofa_jaccard_index + detec_array[x][0][i]
            nsofa = nsofa + 1

        elif detec_array[x][1][i] == "train":
            train_jaccard_index = train_jaccard_index + detec_array[x][0][i]
            ntrain = ntrain + 1

        elif detec_array[x][1][i] == "tvmonitor":
            tvmonitor_jaccard_index = tvmonitor_jaccard_index + detec_array[x][0][i]
            ntvmonitor = ntvmonitor + 1

jaccardIndex_file.close()

jaccardIndex_average_file = open('jaccardIndex_average.txt', 'w')

jaccardIndex_average_file.write("Jaccard Index Average\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("AEROPLANE\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(naeroplane) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(aeroplane_jaccard_index/naeroplane) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("BICYCLE\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(nbicycle) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(bicycle_jaccard_index/nbicycle) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("BIRD\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(nbird) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(bird_jaccard_index/nbird) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("BOAT\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(nboat) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(boat_jaccard_index/nboat) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("BOTTLE\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(nbottle) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(bottle_jaccard_index/nbottle) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("BUS\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(nbus) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(bus_jaccard_index/nbus) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("CAR\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(ncar) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(car_jaccard_index/ncar) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("CAT\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(ncat) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(cat_jaccard_index/ncat) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("CHAIR\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(nchair) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(chair_jaccard_index/nchair) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("COW\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(ncow) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(cow_jaccard_index/ncow) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("DININGTABLE\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(ndiningtable) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(diningtable_jaccard_index/ndiningtable) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("DOG\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(ndog) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(dog_jaccard_index/ndog) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("HORSE\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(nhorse) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(horse_jaccard_index/nhorse) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("MOTORBIKE\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(nmotorbike) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(motorbike_jaccard_index/nmotorbike) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("PERSON\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(nperson) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(person_jaccard_index/nperson) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("POTTEDPLANT\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(npottedplant) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(pottedplant_jaccard_index/npottedplant) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("SHEPP\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(nsheep) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(sheep_jaccard_index/nsheep) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("SOFA\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(nsofa) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(sofa_jaccard_index/nsofa) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("TRAIN\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(ntrain) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(train_jaccard_index/ntrain) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.write("TVMONITOR\n")
jaccardIndex_average_file.write("Numero de veces detectado: " + str(ntvmonitor) + "\n")
jaccardIndex_average_file.write("Jaccard Index Average: " + str(tvmonitor_jaccard_index/ntvmonitor) + "\n")
jaccardIndex_average_file.write("\n")

jaccardIndex_average_file.close()
