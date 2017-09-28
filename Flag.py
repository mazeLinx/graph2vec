import tensorflow as tf


def define_flags():
    #data set
    tf.app.flags.DEFINE_string('dataset_dir', '/Users/jianbinlin/Documents/Jianbin/Code/stru2vec/data/', 'The directory where the dataset files are stored.')
    tf.app.flags.DEFINE_integer('label_size', 39, 'The number of lable size for each sample.')
    tf.app.flags.DEFINE_integer('dense_size', 10, 'The number of dense feature size.')
    tf.app.flags.DEFINE_integer('sample_size', 10400, 'total number of input samples')
    tf.app.flags.DEFINE_integer('node_size', 10400, 'The number of nodes in training data.')

    #model
    tf.app.flags.DEFINE_integer('num_epochs', 2, 'The number of training epoch.')
    tf.app.flags.DEFINE_integer('h1', 128, 'The number of 1st hidden node.')
    tf.app.flags.DEFINE_integer('embedding_size', 10, 'embedding size')
    tf.app.flags.DEFINE_integer('lr', 1e-04, 'init learning rate')
    tf.app.flags.DEFINE_integer('batch_size', 256, 'The number of samples in each batch.')

    return tf.app.flags.FLAGS
