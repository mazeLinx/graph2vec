import numpy as np
import tensorflow as tf


path = r'/Users/jianbinlin/DDNN/project/phone_data/test.tfrecord'

record_iterator = tf.python_io.tf_record_iterator(path=path)


cnt = 0

for string_record in record_iterator:
    
    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    label = (example.features.feature['label'].float_list.value)
    
    #print cnt, label

    cnt += 1

print cnt