import logging
import os.path
import sys
import time
import tensorflow as tf

import model as model

#dataset
tf.app.flags.DEFINE_string('dataset_dir', '/Users/jianbinlin/DDNN/project/stru2vec_data/', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer('label_size', 2, 'The number of lable size for each sample.')
tf.app.flags.DEFINE_integer('fea_size', 4000, 'The number of dense feature size.')
tf.app.flags.DEFINE_integer('sample_size', 201631, 'total number of input samples')
tf.app.flags.DEFINE_integer('node_size', 981749, 'The number of nodes in training data.')
# tf.app.flags.DEFINE_integer('sample_size', 10312, 'total number of input samples')
# tf.app.flags.DEFINE_integer('node_size', 10312, 'The number of nodes in training data.')

#model
tf.app.flags.DEFINE_integer('h1', 12, 'The number of 1st hidden node.')
tf.app.flags.DEFINE_integer('embedding_size', 12, 'embedding size')
tf.app.flags.DEFINE_integer('batch_size', 16, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('num_epochs', 500, 'The number of training epoch.')
tf.app.flags.DEFINE_integer('l2', 0, 'l2 factor')
tf.app.flags.DEFINE_integer('lr', 0.1, 'init learning rate')
tf.app.flags.DEFINE_integer('lr_decay_factor', 0.9, 'learning rate decay ratio')
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 50, 'decay the lr every n epoch')
tf.app.flags.DEFINE_string('lr_decay_type', 'exponential', 'learning rate decay type')

#saver
tf.app.flags.DEFINE_string('check_point', './output/model1/model', 'the directory of checkpoint')
tf.app.flags.DEFINE_integer('init_iter', -1, 'the initial model iteration')


FLAGS = tf.app.flags.FLAGS



def main(_):
    logging_path = "./output/log1002/train_{}.log".format(time.strftime(time.strftime("%m%d-%H_%M_%S", time.localtime())))
    logging.basicConfig(filename=logging_path, level=logging.INFO, format='%(asctime)s %(message)s\t', datefmt='%Y-%m-%d %H:%M:%S')

    for k, v in FLAGS.__flags.iteritems():
        logging.info('{}, {}'.format(k, v))

    global_step = tf.train.get_or_create_global_step()

    # idx, val = model.read_embedding(r'/Users/jianbinlin/DDNN/project/stru2vec_data/feature', [FLAGS.node_size, FLAGS.fea_size])
    # embedding = tf.SparseTensor(indices=idx, values=val, dense_shape=[FLAGS.node_size, FLAGS.fea_size])
    
    label, fea, neig_id, neig_value = model.inputs("train.tfrecord")
    
    logit, loss = model.inference(label, fea, neig_id, neig_value)
    
    lr = model.configure_lr(global_step)
    optimizer = tf.train.AdagradOptimizer(learning_rate=lr) 
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.lr)
    train_op = optimizer.minimize(loss, global_step=global_step)
    
    saver = tf.train.Saver(max_to_keep=FLAGS.num_epochs+1)

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        if FLAGS.init_iter > 0:
            saver.restore(sess, '/Users/jianbinlin/DDNN/project/output/model1/model-{}'.format(FLAGS.init_iter))

        steps_num_per_epoch = int(FLAGS.sample_size / FLAGS.batch_size)
        total_steps = steps_num_per_epoch * FLAGS.num_epochs
        batch_cnt = 0
        epoch_cnt = FLAGS.init_iter + 1
        saver.save(sess, FLAGS.check_point, 0)
        for i in xrange(total_steps):
            res_train, res_logit, res_loss = sess.run([train_op, logit, loss])
            res_lr = sess.run(lr)

            if (i+1) % steps_num_per_epoch == 0:
                saver.save(sess, FLAGS.check_point, epoch_cnt)
                epoch_cnt += 1

            if i % 10000 == 0:
                print(i, epoch_cnt, res_loss)

            logging.info('batch count: {}, epoch count: {}, lr: {:.6f}, loss: {:.6f}'.format(i, epoch_cnt, res_lr, res_loss))


                
        saver.save(sess, FLAGS.check_point, epoch_cnt)

        coord.request_stop()
        
        # gwriter = tf.summary.FileWriter('./output/log1/graph', sess.graph)
        # gwriter.close()
    
    logging.info('done')



if __name__ == '__main__':
    tf.app.run()