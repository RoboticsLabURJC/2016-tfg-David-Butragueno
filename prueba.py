import caffe
import os

model_file = '../examples/mnist/lenet_train_test.prototxt'
petrained_file = '../examples/mnist/lenet_iter_1000.caffemodel'

net = caffe.Classifier(model_file, petrained_file, image_dims=(28,28), raw_scale=255)
score = net.predict([caffe.io.load_image('Cinco.bmp', color=False)], oversample=False)
print score
