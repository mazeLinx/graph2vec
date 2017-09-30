import tensorflow as tf
import numpy as np

tfrecords_filename = '/Users/jianbinlin/DDNN/project/stru2vec_data/test.tfrecord'
record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)


inf = open('out', 'w')
cnt = 0
for string_record in record_iterator:
    # Parse the next example
    example = tf.train.Example()
    example.ParseFromString(string_record)

    # Get the features you stored (change to match your tfrecord writing code)
    label = example.features.feature['label'].float_list.value
    fea_id = example.features.feature['fea_id'].int64_list.value
    fea_value = example.features.feature['fea_value'].float_list.value
    neig_id = example.features.feature['neig_id'].int64_list.value
    neig_value = example.features.feature['neig_value'].float_list.value

    
    #print(cnt, label, fea_id, fea_value, neig_id, neig_value)

    inf.write(' '.join(map(str, neig_id)))
    inf.write('\n')

    cnt += 1


inf.close()