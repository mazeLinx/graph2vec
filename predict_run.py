import logging
import math
import numpy as np
import os.path
import sys
import sklearn.metrics as metrics
import time
import tensorflow as tf

import model as model

#data set
tf.app.flags.DEFINE_string('dataset_dir', '/Users/jianbinlin/DDNN/project/stru2vec_data/garbage_account/', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer('label_size', 2, 'The number of lable size for each sample.')
tf.app.flags.DEFINE_integer('fea_size', 4000, 'The number of dense feature size.')
tf.app.flags.DEFINE_integer('sample_size', 31682, 'total number of input samples')
tf.app.flags.DEFINE_integer('node_size', 981750, 'The number of nodes in training data.')

#model
tf.app.flags.DEFINE_integer('num_epochs', 1, 'The number of training epoch.')
tf.app.flags.DEFINE_integer('h1', 64, 'The number of 1st hidden node.')
tf.app.flags.DEFINE_integer('embedding_size', 12, 'embedding size')
tf.app.flags.DEFINE_integer('batch_size', 1, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('l2', 0.0001, 'l2 factor')


#saver
tf.app.flags.DEFINE_string('check_point', 'model/model', 'the directory of checkpoint')


FLAGS = tf.app.flags.FLAGS


def aucscore(scores):
    y_true = np.array([i[0] for i in scores]).astype(int)
    y_hat = np.array([i[1] for i in scores])

    auc = metrics.roc_auc_score(y_true, y_hat, average=None)

    return max(auc)

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
            # precision = metrics.precision_score(y_true_te_tmp, y_hat_te_tmp, pos_label=1)
            # recall = metrics.recall_score(y_true_te_tmp, y_hat_te_tmp, pos_label=1)
            precision = 0.0
            recall = 0.0
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

def get_para(path):
    res = ""
    with open(path, 'r') as inf:
        for i in inf.readlines():
            if 'batch count:' not in i:
                res += i
            
    return res + '\n'

def main(_):

    logging_path = FLAGS.dataset_dir + "output/log/eval_{}.log".format(time.strftime("%m%d-%H_%M_%S", time.localtime()))
    logging.basicConfig(filename=logging_path, level=logging.INFO, format='%(asctime)s %(message)s\t', datefmt='%Y-%m-%d %H:%M:%S')

    train_log = "/Users/jianbinlin/DDNN/project/stru2vec_data/garbage_account/output/log/train_1221-17_32_10.log"
    logging.info(train_log)
    logging.info(get_para(train_log))

    for iter in np.arange(0, 100, 2):
        with tf.Graph().as_default() as g:
            label, fea, neig_id, neig_value = model.inputs(FLAGS.dataset_dir + "test.tfrecord")

            logit, loss = model.inference(label, fea, neig_id, neig_value)
            
            saver = tf.train.Saver()
        
            cnt = 0
            reslist = []
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
            
                saver.restore(sess, FLAGS.dataset_dir + 'output/model/model-{}'.format(iter))
        
                for i in xrange(FLAGS.sample_size):                                                                                                                                                                                 
                    res_y, res_y_, res_loss = sess.run([label, logit, loss])
                
                    for i in xrange(len(res_y)):
                        reslist.append([res_y[i], res_y_[i], res_loss])                                                   
                 
                coord.request_stop()            
                coord.join(threads)

            loss, f1 = f1score(reslist)
            # loss, f1 = 0, 0
            auc = aucscore(reslist)
            logging.info("epoch: {}, loss:{:.6f}, f1:{:.6f}, auc:{:.6f}".format(iter, loss, f1, auc))
    
    logging.info('done')




if __name__ == '__main__':
    tf.app.run()