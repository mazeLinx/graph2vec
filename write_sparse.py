import numpy as np
import random
import tensorflow as tf


# one MUST randomly shuffle data before putting it into one of these
# formats. Without this, one cannot make use of tensorflow's great
# out of core shuffling.


path = r'/Users/jianbinlin/DDNN/project/phone_data/samples_test.txt'

outpath = "/Users/jianbinlin/DDNN/project/phone_data/test.tfrecord"

writer = tf.python_io.TFRecordWriter(outpath)

cnt = 0
with open(path, 'r') as inf:
    for l in inf.readlines():
        parts = l.split('\t')
        node_id = parts[0]

        fea_id = []
        fea_value = []

        label = map(float, parts[1].split(' '))

        for p in parts[2].split(' '):
            id_v = p.split(':')
            fea_id.append(long(id_v[0]))
            fea_value.append(float(id_v[1]))
        
        neig_id = map(long, parts[3].split(' '))
        neig_value = [1.0 / len(neig_id) for i in neig_id]
        
        if cnt % 1000 == 0:
            print (cnt, node_id, len(label), len(fea_id), len(neig_id))        

        # construct the Example proto boject
        example = tf.train.Example(
            # Example contains a Features proto object
            features=tf.train.Features(
            # Features contains a map of string to Feature proto objects
                feature={
                    # A Feature contains one of either a int64_list, float_list, or bytes_list
                    'label': tf.train.Feature(
                        float_list=tf.train.FloatList(value=label)),
                    'fea_id': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=fea_id)),
                    'fea_value': tf.train.Feature(
                        float_list=tf.train.FloatList(value=fea_value)),
                    'neig_id': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=neig_id)),
                    'neig_value': tf.train.Feature(
                        float_list=tf.train.FloatList(value=neig_value))
            }))

        serialized = example.SerializeToString()
        writer.write(serialized)
        
        cnt += 1



writer.close()

print "done", cnt