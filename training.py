from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import argparse
import os.path
import sys
import math
import time
import tensorflow as tf

from tensorflow.python.ops import variables
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import clip_ops

from tensorflow.python.training import moving_averages
import tensorflow.contrib.layers as core_layers

#dataset
tf.app.flags.DEFINE_string('dataset_dir', '/Users/jianbinlin/DDNN/project/stru2vec/data/', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer('label_size', 39, 'The number of lable size for each sample.')
tf.app.flags.DEFINE_integer('dense_size', 10, 'The number of dense feature size.')
tf.app.flags.DEFINE_integer('sample_size', 6187, 'total number of input samples')
tf.app.flags.DEFINE_integer('node_size', 10400, 'The number of nodes in training data.')

#model
tf.app.flags.DEFINE_integer('h1', 16, 'The number of 1st hidden node.')
tf.app.flags.DEFINE_integer('embedding_size', 10, 'embedding size')

tf.app.flags.DEFINE_integer('batch_size', 32, 'The number of samples in each batch.')


#train
tf.app.flags.DEFINE_integer('num_epochs', 2000, 'The number of training epoch.')
tf.app.flags.DEFINE_integer('l2', 0, 'l2 factor')
tf.app.flags.DEFINE_integer('lr', 0.0002, 'init learning rate')
tf.app.flags.DEFINE_integer('lr_decay_factor', 1, 'learning rate decay ratio')
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 10, 'decay the lr every n epoch')
tf.app.flags.DEFINE_string('lr_decay_type', 'exponential', 'learning rate decay type')

#saver
tf.app.flags.DEFINE_string('check_point', 'model/model', 'the directory of checkpoint')


FLAGS = tf.app.flags.FLAGS

slim = tf.contrib.slim

logging.basicConfig(
        filename="./log/train10_relu_{}.log".format(time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())), 
        level=logging.INFO,
        format='%(asctime)s %(message)s\t',
        datefmt='%Y-%m-%d %H:%M:%S'
        )

def configure_lr(global_step):
    decay_steps = int(FLAGS.sample_size / FLAGS.batch_size * FLAGS.num_epochs_per_decay)

    if FLAGS.lr_decay_type == 'exponential':
        return tf.train.exponential_decay(
                                        FLAGS.lr,
                                        global_step,
                                        decay_steps,
                                        FLAGS.lr_decay_factor,
                                        staircase=True,
                                        name='exponential_decay_learning_rate'
                                        )
    return tf.constant(FLAGS.lr, name='fixed_learning_rate')                                
 

    

def inputs(file_pattern):
    pattern = os.path.join(FLAGS.dataset_dir, file_pattern)
    print(pattern)
    files = tf.gfile.Glob(pattern)
    print(str(files))
    
    capacity = 10000 + 10000 * FLAGS.batch_size
    
    with tf.device('/cpu:0'):
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(files)
            reader = tf.TFRecordReader()
            key, value = reader.read_up_to(filename_queue, FLAGS.batch_size)

            record = tf.train.shuffle_batch([value], batch_size=FLAGS.batch_size, num_threads=2, min_after_dequeue=FLAGS.batch_size * 1000, capacity=capacity, enqueue_many=True)  

            parsed = tf.parse_example(
                record,
                features={
                'label' : tf.FixedLenFeature([FLAGS.label_size], dtype=tf.float32),
                'dense' : tf.FixedLenFeature([FLAGS.dense_size], dtype=tf.float32),
                'sparse_id' : tf.VarLenFeature(dtype=tf.int64),
                'sparse_value' : tf.VarLenFeature(dtype=tf.float32),
             })
                
    return parsed['label'], parsed['dense'], parsed['sparse_id'], parsed['sparse_value']

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
    
    # w_dense_l2 = tf.nn.l2_loss(w_dense)
    # w1_l2 = tf.nn.l2_loss(w1)
    # loss_l2 = tf.reduce_sum(FLAGS.l2 * w1_l2 + FLAGS.l2 * w_dense_l2 + loss)

    return logit, loss

def inference_nodense(label, dense, sparse_id, sparse_value):
    embedding_store = tf.Variable(tf.truncated_normal([FLAGS.node_size, FLAGS.embedding_size], stddev = 0.5))
    w1 = tf.Variable(tf.truncated_normal([FLAGS.embedding_size, FLAGS.label_size], stddev = 0.5))

    embedding = tf.nn.embedding_lookup_sparse(embedding_store, sparse_id, sparse_value, combiner='sum')

    emb_act = tf.nn.relu(embedding)

    logit = tf.matmul(emb_act, w1)
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
    
    return logit, loss


def train(label, dense, sparse_id, sparse_value, global_step):
    logit, loss = inference_nodense(label, dense, sparse_id, sparse_value)

    #lr = configure_lr(global_step)
    #logging.info("lr is {}".format(lr))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.lr)
    train_op = optimizer.minimize(loss)

    return train_op, logit, loss

def inference_bn(label, dense, sparse_id, sparse_value):
    embedding_store = tf.Variable(tf.truncated_normal([FLAGS.node_size, FLAGS.embedding_size], stddev = 0.5))
    w_dense = tf.Variable(tf.truncated_normal([FLAGS.dense_size, FLAGS.embedding_size], stddev = 0.5))
    w1 = tf.Variable(tf.truncated_normal([FLAGS.embedding_size, FLAGS.label_size], stddev = 0.5))

    embedding = tf.nn.embedding_lookup_sparse(embedding_store, sparse_id, sparse_value, combiner='sum')

    dense_x_w = tf.matmul(dense, w_dense) 
    fea_p_emb = tf.add(embedding, dense_x_w)
    
    act = tf.nn.relu(fea_p_emb)
    bn = tf.layers.batch_normalization(act, training=True)

    logit = tf.matmul(bn, w1)
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
    
    return logit, loss


def train_bn(label, dense, sparse_id, sparse_value):
    logit, loss = inference_bn(label, dense, sparse_id, sparse_value)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.lr)
    
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss) 

    return train_op, logit, loss




def main(_):
    
    for k, v in FLAGS.__flags.iteritems():
        logging.info('{}, {}'.format(k, v))

    global_step = slim.get_or_create_global_step()
    label, dense, sparse_id, sparse_value = inputs("train.tfrecord")

    train_op, logit, loss = train(label, dense, sparse_id, sparse_value, global_step)

    saver = tf.train.Saver(max_to_keep=FLAGS.num_epochs+1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        steps_num_per_epoch = int(FLAGS.sample_size / FLAGS.batch_size)
        total_steps = steps_num_per_epoch * FLAGS.num_epochs
        batch_cnt = 0
        epoch_cnt = 0

        for i in xrange(total_steps):    
                                                                                                                                                                            
            res_train, res_logit, res_loss = sess.run([train_op, logit, loss])
            
            if i % 100 == 0:
                print(i, epoch_cnt, res_loss)

            logging.info('batch count: {}, epoch count: {}, loss: {}'.format(i, epoch_cnt, res_loss))

            if i % steps_num_per_epoch == 0:
                print("------")                
                saver.save(sess, FLAGS.check_point, epoch_cnt)
                epoch_cnt += 1
                
        saver.save(sess, FLAGS.check_point, epoch_cnt)

        coord.request_stop()
        
        gwriter = tf.summary.FileWriter('./log/graph', sess.graph)
        gwriter.close()
    
    logging.info('done')

if __name__ == '__main__':
    tf.app.run()