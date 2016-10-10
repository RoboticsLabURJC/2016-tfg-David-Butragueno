import caffe
import lmdb

lmdb_env = lmdb.open('/home/davidbutra/Escritorio/caffe/examples/mnist/mnist_train_lmdb')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

Zero = 0
One = 0
Two = 0
Three = 0
Four = 0
Five = 0
Six = 0
Seven = 0
Eight = 0
Nine = 0
TotalImages = 0

for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    print label

    if label == 0:
        Zero = Zero + 1
    elif label == 1:
        One = One + 1
    elif label == 2:
        Two = Two + 1
    elif label == 3:
        Three = Three + 1
    elif label == 4:
        Four = Four + 1
    elif label == 5:
        Five = Five + 1
    elif label == 6:
        Six = Six + 1
    elif label == 7:
       Seven = Seven + 1
    elif label == 8:
        Eight = Eight + 1
    elif label == 9:
        Nine = Nine + 1

labelfile=open('label.txt', 'w')

labelfile.write("Zero: ")
labelfile.write("%s\n" % (Zero))

labelfile.write("One: ")
labelfile.write("%s\n" % (One))

labelfile.write("Two: ")
labelfile.write("%s\n" % (Two))

labelfile.write("Three: ")
labelfile.write("%s\n" % (Three))

labelfile.write("Four: ")
labelfile.write("%s\n" % (Four))

labelfile.write("Five: ")
labelfile.write("%s\n" % (Five))

labelfile.write("Six: ")
labelfile.write("%s\n" % (Six))

labelfile.write("Seven: ")
labelfile.write("%s\n" % (Seven))

labelfile.write("Eight: ")
labelfile.write("%s\n" % (Eight))

labelfile.write("Nine: ")
labelfile.write("%s\n" % (Nine))

TotalImages = Zero + One + Two + Three + Four + Five + Six + Seven + Eight + Nine

labelfile.write("Total images: ")
labelfile.write("%s\n" % (TotalImages))
