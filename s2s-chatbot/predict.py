import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import tensorflow as tf
import numpy as np
import time
from data.twitter import data
metadata, idx_q, idx_a = data.load_data(PATH='data/twitter/')

def predict(question):
    batch_size=32
    xvocab_size = len(metadata['idx2w']) 
    emb_dim = 1024

    w2idx = metadata['w2idx']   
    idx2w = metadata['idx2w']   

    unk_id = w2idx['unk']   
    pad_id = w2idx['_']     

    start_id = xvocab_size  
    end_id = xvocab_size+1  

    w2idx.update({'start_id': start_id})
    w2idx.update({'end_id': end_id})
    idx2w = idx2w + ['start_id', 'end_id']

    xvocab_size = yvocab_size = xvocab_size + 2

    
    encode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="encode_seqs")
    decode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="decode_seqs")
    target_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_seqs")
    target_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_mask")
    net_out, _ = model(encode_seqs, decode_seqs,xvocab_size,emb_dim,is_train=True, reuse=False)

    encode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs")
    decode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_seqs")
    net, net_rnn = model(encode_seqs2, decode_seqs2,xvocab_size,emb_dim,is_train=False, reuse=True)
    y = tf.nn.softmax(net.outputs)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name='n.npz', network=net)

    seed=question
    try:
        seed_id = [w2idx[w] for w in seed.split(" ")]
        for _ in range(1):
            state = sess.run(net_rnn.final_state_encode,
                            {encode_seqs2: [seed_id]})
            o, state = sess.run([y, net_rnn.final_state_decode],
                            {net_rnn.initial_state_decode: state,
                            decode_seqs2: [[start_id]]})
            w_id = tl.nlp.sample_top(o[0], top_k=1)
            w = idx2w[w_id]
            sentence = [w]
            for _ in range(20):
                o, state = sess.run([y, net_rnn.final_state_decode],
                                {net_rnn.initial_state_decode: state,
                                decode_seqs2: [[w_id]]})
                w_id = tl.nlp.sample_top(o[0], top_k=0)
                w = idx2w[w_id]
                if w_id == end_id:
                    break
                sentence = sentence + [w]
            a=' '.join(sentence)
            return (a)
    except:
        return "I'm sorry sir, I did not understand your query, please try again using keywords or check my documentation by typing 'docs'."

def model(encode_seqs, decode_seqs,xvocab_size,emb_dim,is_train=True, reuse=False):
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("embedding") as vs:
                net_encode = EmbeddingInputlayer(inputs = encode_seqs,vocabulary_size = xvocab_size,embedding_size = emb_dim,name = 'seq_embedding')
                vs.reuse_variables()
                tl.layers.set_name_reuse(True)
                net_decode = EmbeddingInputlayer(inputs = decode_seqs,vocabulary_size = xvocab_size,embedding_size = emb_dim,name = 'seq_embedding')
            net_rnn = Seq2Seq(net_encode, net_decode,
                    cell_fn = tf.contrib.rnn.BasicLSTMCell,
                    n_hidden = emb_dim,
                    initializer = tf.random_uniform_initializer(-0.1, 0.1),
                    encode_sequence_length = retrieve_seq_length_op2(encode_seqs),
                    decode_sequence_length = retrieve_seq_length_op2(decode_seqs),
                    initial_state_encode = None,
                    dropout = (0.5 if is_train else None),
                    n_layer = 3,
                    return_seq_2d = True,
                    name = 'seq2seq')
            net_out = DenseLayer(net_rnn, n_units=xvocab_size, act=tf.identity, name='output')
        return net_out, net_rnn
#Credits to tensorlayer for parts of the model.
