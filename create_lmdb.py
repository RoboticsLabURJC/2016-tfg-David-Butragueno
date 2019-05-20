import lmdb
import caffe

lmdb_env_voc = lmdb.open("/home/docker/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb")
lmdb_txn_voc = lmdb_env_voc.begin()
lmdb_cursor_voc = lmdb_txn_voc.cursor()
datum_voc = caffe.proto.caffe_pb2.Datum()

lmdb_env_coco = lmdb.open("/home/docker/data/coco/lmdb/coco_train_lmdb")
lmdb_txn_coco = lmdb_env_coco.begin()
lmdb_cursor_coco = lmdb_txn_coco.cursor()
datum_coco = caffe.proto.caffe_pb2.Datum()

new_lmdb_env = lmdb.open("/home/docker/data/custom_databases/databases",map_size=int(1e12))
new_lmdb_txn = new_lmdb_env.begin(write=True)
new_lmdb_cursor = new_lmdb_txn.cursor()

batch_size = 256
item_id = -1

for key, value in lmdb_cursor_voc:
    item_id = item_id + 1   

    datum_voc.ParseFromString(value)

    keystr = "{:0>8d}".format(item_id)
    new_lmdb_txn.put(keystr, datum_voc.SerializeToString())

    if(item_id + 1) % batch_size == 0:
        new_lmdb_txn.commit()
        new_lmdb_txn = new_lmdb_env.begin(write=True)
        print(item_id + 1)

if(item_id+1) % batch_size != 0:
    new_lmdb_txn.commit()
    print "last_batch"

batch_size=256
item_id=-1
print lmdb_env_coco.stat()
for (key, value) in lmdb_cursor_coco:
    item_id = item_id + 1
    new_lmdb_cursor.put(key, value)

    if(item_id + 1) % batch_size == 0:
        new_lmdb_txn.commit()
        new_lmdb_txn = new_lmdb_env.begin(write=True)
        print(item_id + 1)

if(item_id+1) % batch_size != 0:
    new_lmdb_txn.commit()
    print "last_batch_coco"
