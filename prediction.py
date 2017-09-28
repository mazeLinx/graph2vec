
import argparse
import csv
import logging
import math
import numpy as np
import os.path
import sys
import sklearn.metrics as metrics
import time
import tensorflow as tf

#data set
tf.app.flags.DEFINE_string('dataset_dir', '/Users/jianbinlin/DDNN/project/stru2vec/data/', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer('label_size', 39, 'The number of lable size for each sample.')
tf.app.flags.DEFINE_integer('dense_size', 10, 'The number of dense feature size.')
tf.app.flags.DEFINE_integer('sample_size', 10400, 'total number of input samples')
tf.app.flags.DEFINE_integer('node_size', 10400, 'The number of nodes in training data.')

#model
tf.app.flags.DEFINE_integer('num_epochs', 1, 'The number of training epoch.')
tf.app.flags.DEFINE_integer('h1', 128, 'The number of 1st hidden node.')
tf.app.flags.DEFINE_integer('embedding_size', 10, 'embedding size')
tf.app.flags.DEFINE_integer('batch_size', 1, 'The number of samples in each batch.')


#saver
tf.app.flags.DEFINE_string('check_point', 'model/model', 'the directory of checkpoint')


FLAGS = tf.app.flags.FLAGS

def sigmoid(x):
    return 1/(1+np.exp(-x))

def inputs(file_pattern):
    pattern = os.path.join(FLAGS.dataset_dir, file_pattern)
    print(pattern)
    files = tf.gfile.Glob(pattern)
    print(str(files))
    
    capacity = 10000 + 10000 * FLAGS.batch_size
    
    with tf.device('/cpu:0'):
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(files, num_epochs = FLAGS.num_epochs)
            reader = tf.TFRecordReader()
            key, value = reader.read_up_to(filename_queue, FLAGS.batch_size)

            record = tf.train.batch([value], batch_size=FLAGS.batch_size, capacity=capacity, enqueue_many=True)  

            parsed = tf.parse_example(
                record,
                features={
                'label' : tf.FixedLenFeature([FLAGS.label_size], dtype=tf.float32),
                'dense' : tf.FixedLenFeature([FLAGS.dense_size], dtype=tf.float32),
                'sparse_id' : tf.VarLenFeature(dtype=tf.int64),
                'sparse_value' : tf.VarLenFeature(dtype=tf.float32),
             })
                
    return parsed['label'], parsed['dense'], parsed['sparse_id'], parsed['sparse_value']



def f1score(scores):
    test_f1 = 0.0
    y_true_te = np.array([i[0] for i in scores])
    y_hat_te = np.array([i[1] for i in scores])
    loss = np.array([i[2] for i in scores])
    sample_size = len(loss)  

    for j in range(FLAGS.label_size):
        best_te_f1=0
        best_thresh=0
        best_pre = 0
        best_pre_rec = 0
        best_rec = 0
        cur_pre = 0
        cur_rec = 0
        for thresh in np.arange(40, 100, 2):
            thresh_te = np.percentile(y_hat_te[:,j], thresh)
            y_hat_te_tmp = (y_hat_te[:,j]>=thresh_te).astype(int)
            y_true_te_tmp = y_true_te[:,j].astype(int)
            
            te_f1 = metrics.f1_score(y_true_te_tmp, y_hat_te_tmp, pos_label=1)
            precision = metrics.precision_score(y_true_te_tmp, y_hat_te_tmp, pos_label=1)
            recall = metrics.recall_score(y_true_te_tmp, y_hat_te_tmp, pos_label=1)
            if te_f1 > best_te_f1:
                best_te_f1 = te_f1
                best_thresh = thresh
                cur_pre = precision
                cur_rec = recall
            if best_pre < precision:
                best_pre = precision
                best_pre_rec = recall
            if best_rec < recall:
                best_rec = recall

        test_f1 += best_te_f1

    test_f1 = test_f1 / FLAGS.label_size   
    ave_loss = sum(loss) / sample_size

    return ave_loss, test_f1

def inference(label, dense, sparse_id, sparse_value):
    embedding_store = tf.Variable(tf.truncated_normal([FLAGS.node_size, FLAGS.embedding_size], stddev = 0.5))
    w_dense = tf.Variable(tf.truncated_normal([FLAGS.dense_size, FLAGS.embedding_size], stddev = 0.5))
    w1 = tf.Variable(tf.truncated_normal([FLAGS.embedding_size, FLAGS.label_size], stddev = 0.5))

    embedding = tf.nn.embedding_lookup_sparse(embedding_store, sparse_id, sparse_value, combiner='sum')

    dense_x_w = tf.matmul(dense, w_dense)  
    fea_p_emb = tf.add(embedding, dense_x_w) 
    fea_p_emb_act = tf.nn.relu(fea_p_emb)

    logit = tf.matmul(fea_p_emb_act, w1)
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
    
    return logit, loss

def inference_nodense(label, dense, sparse_id, sparse_value):
    embedding_store = tf.Variable(tf.truncated_normal([FLAGS.node_size, FLAGS.embedding_size], stddev = 0.5))
    w1 = tf.Variable(tf.truncated_normal([FLAGS.embedding_size, FLAGS.label_size], stddev = 0.5))

    embedding = tf.nn.embedding_lookup_sparse(embedding_store, sparse_id, sparse_value, combiner='sum')

    emb_act = tf.nn.relu(embedding)

    logit = tf.matmul(emb_act, w1)
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
    
    return logit, loss

def inference_bn(label, dense, sparse_id, sparse_value):
    embedding_store = tf.Variable(tf.truncated_normal([FLAGS.node_size, FLAGS.embedding_size], stddev = 0.5))
    w_dense = tf.Variable(tf.truncated_normal([FLAGS.dense_size, FLAGS.embedding_size], stddev = 0.5))
    w1 = tf.Variable(tf.truncated_normal([FLAGS.embedding_size, FLAGS.label_size], stddev = 0.5))

    embedding = tf.nn.embedding_lookup_sparse(embedding_store, sparse_id, sparse_value, combiner='sum')

    dense_x_w = tf.matmul(dense, w_dense) 
    fea_p_emb = tf.add(embedding, dense_x_w)
    
    act = tf.nn.relu(fea_p_emb)
    bn = tf.layers.batch_normalization(act, training=False)

    logit = tf.matmul(bn, w1)
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
    
    return logit, loss
    

def get_para(path):
    res = ""
    with open(path, 'r') as inf:
        for i in inf.readlines():
            if 'batch count:' not in i:
                res += i
            
    return res + '\n'

def main(_):

    logging.basicConfig(
        filename="./log/eval10_dense_{}.log".format(time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())), 
        level=logging.INFO,
        format='%(asctime)s %(message)s\t',
        datefmt='%Y-%m-%d %H:%M:%S'
        )

    train_log = "/Users/jianbinlin/DDNN/project/stru2vec_online/log/train10_relu_2017-09-27_13_54_16.log"
    logging.info(train_log)
    logging.info(get_para(train_log))

    for iter in np.arange(0, 2001, 100):
        with tf.Graph().as_default() as g:
            label, dense, sparse_id, sparse_value = inputs("test.tfrecord")

            logit, loss = inference_nodense(label, dense, sparse_id, sparse_value)
            
            saver = tf.train.Saver()
        
            cnt = 0
            reslist = []
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
            
                saver.restore(sess, '/Users/jianbinlin/DDNN/project/stru2vec_online/model/model-{}'.format(iter))
        
                try:    
                    while not coord.should_stop():                                                                                                                                                                             
                        res_y, res_y_, res_loss = sess.run([label, logit, loss])
                
                        for i in xrange(len(res_y)):
                            reslist.append([res_y[i], res_y_[i], res_loss])                    
                
                except tf.errors.OutOfRangeError:
                    print("prediction done")

                finally:    
                    coord.request_stop()
            
                coord.join(threads)

            loss, f1 = f1score(reslist)
            logging.info("epoch: {}, loss:{}, f1:{}".format(iter, loss, f1))
            print(iter, loss, f1)
    
    logging.info('done')


if __name__ == '__main__':
    tf.app.run()