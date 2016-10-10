import caffe
import lmdb

lmdb_env = lmdb.open('/home/davidbutra/Escritorio/caffe/examples/mnist/mnist_train_lmdb')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()
t = 0
n = 0
i = 0

for key, value in lmdb_cursor:
    #if t == 0:
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    #print label
    #print data
    t = t + 1

width = 28
height = 28
ftype = 'P2'

pgmfile=open('data.pgm', 'w')
pgmfile.write("%s\n" % (ftype))
pgmfile.write("%d %d\n" % (width,height))
pgmfile.write("255\n")

txtfile=open('data.txt', 'w')
txtfile.write("%s\n" % (ftype))
txtfile.write("%d %d\n" % (width,height))
txtfile.write("255\n")

while i < height:
    if n == width - 1:
        pgmfile.write("%s\n" % (data[0][i][n]))
        txtfile.write("%s\n" % (data[0][i][n]))
        i = i + 1
        n = 0   
    elif data[0][i][n + 1] < 10:
        pgmfile.write("%s   " % (data[0][i][n]))
        txtfile.write("%s   " % (data[0][i][n]))
        n = n + 1
    elif data[0][i][n + 1] < 100:
        pgmfile.write("%s  " % (data[0][i][n]))
        txtfile.write("%s  " % (data[0][i][n]))
        n = n + 1
    else:
        pgmfile.write("%s " % (data[0][i][n]))
        txtfile.write("%s " % (data[0][i][n]))
        n = n + 1

pgmfile.close()
