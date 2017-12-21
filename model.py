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
        
    capacity = 10000 + 10000 * FLAGS.batch_size
    
    with tf.device('/cpu:0'):
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(files)
            reader = tf.TFRecordReader()
            key, value = reader.read_up_to(filename_queue, 10240)

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
    embedding_store = tf.Variable(tf.random_uniform([FLAGS.node_size, FLAGS.embedding_size],  -0.1, 0.1))
    w_fea = tf.Variable(tf.random_uniform([FLAGS.fea_size, FLAGS.embedding_size], -0.5, 0.5))
    w1 = tf.Variable(tf.random_uniform([FLAGS.embedding_size, FLAGS.label_size], -0.5, 0.5))

    embedding = tf.nn.embedding_lookup_sparse(embedding_store, neig_id, neig_value, combiner='sum')

    fea_layer= tf.sparse_tensor_dense_matmul(fea, w_fea)
    h1_layer = tf.add(embedding, fea_layer)
    # h1_layer_act = tf.nn.relu(h1_layer)
    h1_layer_act = tf.nn.sigmoid(h1_layer)
    # h1_layer_act = tf.nn.tanh(h1_layer)
    logit = tf.matmul(h1_layer_act, w1)
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
    
    return logit, loss

def inference_nodense(label, fea, neig_id, neig_value):
    embedding_store = tf.Variable(tf.random_uniform([FLAGS.node_size, FLAGS.embedding_size],  -0.1, 0.1))
    w1 = tf.Variable(tf.random_uniform([FLAGS.embedding_size, FLAGS.label_size], -0.5, 0.5))

    embedding = tf.nn.embedding_lookup_sparse(embedding_store, neig_id, neig_value, combiner='sum')
    
    h1_layer_act = tf.nn.sigmoid(embedding)
    logit = tf.matmul(h1_layer_act, w1)
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
    
    return logit, loss

def inference_feaact(label, fea, neig_id, neig_value):
    embedding_store = tf.Variable(tf.random_uniform([FLAGS.node_size, FLAGS.embedding_size],  -0.5, 0.5))
    w_fea = tf.Variable(tf.random_uniform([FLAGS.fea_size, FLAGS.embedding_size], -0.5, 0.5))
    w1 = tf.Variable(tf.random_uniform([FLAGS.embedding_size, FLAGS.label_size], -0.5, 0.5))

    embedding = tf.nn.embedding_lookup_sparse(embedding_store, neig_id, neig_value, combiner='sum')

    fea_layer= tf.sparse_tensor_dense_matmul(fea, w_fea)
    fea_layer_act = tf.nn.relu(fea_layer)
    h1_layer = tf.add(embedding, fea_layer_act)
    # h1_layer_act = tf.nn.relu(h1_layer)
    h1_layer_act = tf.nn.sigmoid(h1_layer)
    # h1_layer_act = tf.nn.tanh(h1_layer)
    logit = tf.matmul(h1_layer_act, w1)
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
    
    return logit, loss

def read_embedding(path, shape):
    idx = []
    val = []
    with open(path, 'r') as f:
        for i, l in enumerate(f.readlines()):
            ex = l.strip().split()[1:]
          
            idx.extend([[i, int(tmp.split(':')[0])] for tmp in ex])
            
            vs = [float(tmp.split(':')[1]) for tmp in ex]
            val.extend(vs)
        
    return idx, val

def inference_pretrain(label, fea, embedding, neig_id, neig_value):
    embedding_store = tf.Variable(embedding,  trainable=False)
    w_emb = tf.Variable(tf.random_uniform([FLAGS.embedding_size, FLAGS.h1], -0.5, 0.5))
    w_fea = tf.Variable(tf.random_uniform([FLAGS.fea_size, FLAGS.h1], -0.5, 0.5))
    w1 = tf.Variable(tf.random_uniform([FLAGS.h1, FLAGS.label_size], -0.5, 0.5))

    embedding = tf.nn.embedding_lookup_sparse(embedding_store, neig_id, neig_value, combiner='sum')
    embedding_layer = tf.sparse_tensor_dense_matmul(embedding, w_emb)

    fea_layer= tf.sparse_tensor_dense_matmul(fea, w_fea)
  
    h1_layer = tf.add(embedding_layer, fea_layer)
    h1_layer_act = tf.nn.sigmoid(h1_layer)

    logit = tf.matmul(h1_layer_act, w1)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
    
    return logit, loss

def inference_bn(label, dense, sparse_id, sparse_value):
    embedding_store = tf.Variable(tf.truncated_normal([FLAGS.node_size, FLAGS.embedding_size], stddev = 0.5))
    w_dense = tf.Variable(tf.truncated_normal([FLAGS.fea_size, FLAGS.embedding_size], stddev = 0.5))
    w1 = tf.Variable(tf.truncated_normal([FLAGS.embedding_size, FLAGS.label_size], stddev = 0.5))

    embedding = tf.nn.embedding_lookup_sparse(embedding_store, sparse_id, sparse_value, combiner='sum')

    dense_x_w = tf.matmul(dense, w_dense) 
    fea_p_emb = tf.add(embedding, dense_x_w)
    
    act = tf.nn.relu(fea_p_emb)
    bn = tf.layers.batch_normalization(act, training=True)

    logit = tf.matmul(bn, w1)
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
    
    return logit, loss


def inference_l2(label, fea, neig_id, neig_value):
    embedding_store = tf.Variable(tf.random_uniform([FLAGS.node_size, FLAGS.embedding_size],  -0.5, 0.5))
    w_fea = tf.Variable(tf.random_uniform([FLAGS.fea_size, FLAGS.embedding_size], -0.5, 0.5))
    w1 = tf.Variable(tf.random_uniform([FLAGS.embedding_size, FLAGS.label_size], -0.5, 0.5))

    embedding = tf.nn.embedding_lookup_sparse(embedding_store, neig_id, neig_value, combiner='sum')

    fea_layer= tf.sparse_tensor_dense_matmul(fea, w_fea)
    h1_layer = tf.add(embedding, fea_layer)
    h1_layer_act = tf.nn.sigmoid(h1_layer)
    logit = tf.matmul(h1_layer_act, w1)   

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))

    #l2
    reg_w1 = tf.nn.l2_loss(w1)
    reg_w_fea = tf.nn.l2_loss(w_fea)
    reg_loss = tf.reduce_sum(loss + FLAGS.l2*reg_w1 + FLAGS.l2*reg_w_fea)
    return logit, reg_loss

def inference_init(label, fea, neig_id, neig_value):
    embedding_store = tf.Variable(tf.random_uniform([FLAGS.node_size, FLAGS.embedding_size],  -0.1, 0.1))
    w_fea = tf.get_variable('w_fea', [FLAGS.fea_size, FLAGS.embedding_size], initializer=tf.contrib.layers.xavier_initializer())
    w1 = tf.get_variable('w1', [FLAGS.embedding_size, FLAGS.label_size], initializer=tf.contrib.layers.xavier_initializer())

    embedding = tf.nn.embedding_lookup_sparse(embedding_store, neig_id, neig_value, combiner='sum')

    fea_layer= tf.sparse_tensor_dense_matmul(fea, w_fea)
    h1_layer = tf.add(embedding, fea_layer)
    h1_layer_act = tf.nn.sigmoid(h1_layer)
    logit = tf.matmul(h1_layer_act, w1)
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
    
    return logit, loss