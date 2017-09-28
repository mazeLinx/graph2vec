




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

            record = tf.train.shuffle_batch([value], batch_size=FLAGS.batch_size, min_after_dequeue=FLAGS.batch_size * 1000, capacity=capacity, enqueue_many=True)  

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
    embedding_store = tf.Variable(tf.truncated_normal([FLAGS.node_size, FLAGS.embedding_size], stddev = 0.01))
    w_dense = tf.Variable(tf.truncated_normal([FLAGS.dense_size, FLAGS.embedding_size], stddev = 0.01))
    w1 = tf.Variable(tf.truncated_normal([FLAGS.embedding_size, FLAGS.label_size], stddev = 0.01))

    embedding = tf.nn.embedding_lookup_sparse(embedding_store, sparse_id, sparse_value, combiner='sum')

    print(w1.name)

    dense_x_w = tf.matmul(dense, w_dense)   #n * esize
    fea_p_emb = tf.add(embedding, dense_x_w) #
    logit = tf.matmul(fea_p_emb, w1)
    
    loss = tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=logit, logits=label), 0))
    
    return logit, loss