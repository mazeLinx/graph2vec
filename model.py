import logging
import argparse
import os.path
import sys
import math
import time
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


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
            key, value = reader.read_up_to(filename_queue, FLAGS.batch_size * 100)

            record = tf.train.shuffle_batch([value], batch_size=FLAGS.batch_size, num_threads=2, min_after_dequeue=FLAGS.batch_size * 1000, capacity=capacity, enqueue_many=True)  

            parsed = tf.parse_example(
                record,
                features={
                'label' : tf.FixedLenFeature([FLAGS.label_size], dtype=tf.float32),
                "feature": tf.SparseFeature(index_key="fea_id", value_key="fea_value", dtype=tf.float32, size=FLAGS.fea_size),
                'neig_id' : tf.VarLenFeature(dtype=tf.int64),
                'neig_value' : tf.VarLenFeature(dtype=tf.float32),
             })
                
    return parsed['label'], parsed['feature'], parsed['neig_id'], parsed['neig_value']


def inference(label, fea, neig_id, neig_value):
    embedding_store = tf.Variable(tf.random_uniform([FLAGS.node_size, FLAGS.embedding_size],  -0.5, 0.5))
    w_fea = tf.Variable(tf.random_uniform([FLAGS.fea_size, FLAGS.embedding_size], -0.5, 0.5))
    w1 = tf.Variable(tf.random_uniform([FLAGS.embedding_size, FLAGS.label_size], -0.5, 0.5))


    embedding = tf.nn.embedding_lookup_sparse(embedding_store, neig_id, neig_value, combiner='sum')

    fea_layer= tf.sparse_tensor_dense_matmul(fea, w_fea)
    h1_layer = tf.add(embedding, fea_layer)
    h1_layer_act = tf.nn.relu(h1_layer)

    logit = tf.matmul(h1_layer_act, w1)
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
    
    return logit, loss