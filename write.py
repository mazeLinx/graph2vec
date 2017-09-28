import numpy as np
import tensorflow as tf


# one MUST randomly shuffle data before putting it into one of these
# formats. Without this, one cannot make use of tensorflow's great
# out of core shuffling.
tf.app.flags.DEFINE_integer('label_size', 39, 'The number of lable size for each sample.')
tf.app.flags.DEFINE_integer('dense_size', 128, 'The number of feature size for each node.')

FLAGS = tf.app.flags.FLAGS


path = r'/Users/jianbinlin/DDNN/project/stru2vec/data/samples128_test.txt'

writer = tf.python_io.TFRecordWriter("/Users/jianbinlin/DDNN/project/stru2vec/data/test128.tfrecord")


cnt = 0
with open(path, 'r') as inf:
    for l in inf.readlines():
        parts = l.split(' ')
        node_id = parts[0]
        label = map(float, parts[1 : 1 + FLAGS.label_size])
        dense = map(float, parts[1 + FLAGS.label_size : 1 + FLAGS.label_size + FLAGS.dense_size])
        sparse_id = map(long, parts[1 + FLAGS.label_size + FLAGS.dense_size : ])
        sparse_value = [1.0 / len(sparse_id) for i in sparse_id]

        print (cnt, node_id, len(label), len(dense), len(sparse_id))
        cnt += 1

        # construct the Example proto boject
        example = tf.train.Example(
            # Example contains a Features proto object
            features=tf.train.Features(
            # Features contains a map of string to Feature proto objects
                feature={
                    # A Feature contains one of either a int64_list, float_list, or bytes_list
                    'label': tf.train.Feature(
                        float_list=tf.train.FloatList(value=label)),
                    'dense': tf.train.Feature(
                        float_list=tf.train.FloatList(value=dense)),
                    'sparse_id': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=sparse_id)),
                    'sparse_value': tf.train.Feature(
                        float_list=tf.train.FloatList(value=sparse_value))
            }))
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)

