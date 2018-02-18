import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import tensorflow as tf
import numpy as np
import time
from data.twitter import data
metadata, idx_q, idx_a = data.load_data(PATH='data/twitter/')  # Twitter
(trainX, trainY), (testX, testY), (validX, validY) = data.split_dataset(idx_q, idx_a)

trainX = trainX.tolist()
trainY = trainY.tolist()
testX = testX.tolist()
testY = testY.tolist()
validX = validX.tolist()
validY = validY.tolist()

trainX = tl.prepro.remove_pad_sequences(trainX)
trainY = tl.prepro.remove_pad_sequences(trainY)
testX = tl.prepro.remove_pad_sequences(testX)
testY = tl.prepro.remove_pad_sequences(testY)
validX = tl.prepro.remove_pad_sequences(validX)
validY = tl.prepro.remove_pad_sequences(validY)

xseq_len = len(trainX)
yseq_len = len(trainY)
assert xseq_len == yseq_len
batch_size = 32
n_step = int(xseq_len/batch_size)
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

target_seqs = tl.prepro.sequences_add_end_id([trainY[10]], end_id=end_id)[0]
decode_seqs = tl.prepro.sequences_add_start_id([trainY[10]], start_id=start_id, remove_last=False)[0]
target_mask = tl.prepro.sequences_get_mask([target_seqs])[0]

def model(encode_seqs, decode_seqs, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        with tf.variable_scope("embedding") as vs:
            net_encode = EmbeddingInputlayer(
                inputs = encode_seqs,
                vocabulary_size = xvocab_size,
                embedding_size = emb_dim,
                name = 'seq_embedding')
            vs.reuse_variables()
            tl.layers.set_name_reuse(True)
            net_decode = EmbeddingInputlayer(
                inputs = decode_seqs,
                vocabulary_size = xvocab_size,
                embedding_size = emb_dim,
                name = 'seq_embedding')
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

encode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="encode_seqs")
decode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="decode_seqs")
target_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_seqs")
target_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_mask") 
net_out, net_rnn = model(encode_seqs, decode_seqs, is_train=True, reuse=False)

loss = tl.cost.cross_entropy_seq_with_mask(logits=net_out.outputs, target_seqs=target_seqs, input_mask=target_mask, return_details=False, name='cost')

net_out.print_params(False)

lr = 0.0001
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
saver = tf.train.Saver()

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
sess.run(tf.global_variables_initializer())

print("Training loop started @ {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
epoch_start_time = time.time()
n_epoch = 50
for epoch in range(n_epoch):
    epoch_time = time.time()
    from sklearn.utils import shuffle
    trainX, trainY = shuffle(trainX, trainY, random_state=0)
    total_err, n_iter = 0, 0
    for X, Y in tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=batch_size, shuffle=False):
        step_time = time.time()

        X = tl.prepro.pad_sequences(X)
        _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=end_id)
        _target_seqs = tl.prepro.pad_sequences(_target_seqs)

        _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=False)
        _decode_seqs = tl.prepro.pad_sequences(_decode_seqs)
        _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

        _, err = sess.run([train_op, loss],
                        {encode_seqs: X,
                        decode_seqs: _decode_seqs,
                        target_seqs: _target_seqs,
                        target_mask: _target_mask})

        print("Epoch[%d/%d] step:[%d/%d] loss:%f took:%.5fs" % (epoch, n_epoch, n_iter, n_step, err, time.time() - step_time))

        total_err += err; n_iter += 1
        print("Epoch[%d/%d] averaged loss:%f took:%.5fs" % (epoch, n_epoch, total_err/n_iter, time.time()-epoch_time))
        a=input('Continue TRaining?[y/n]: ')
        if a=='y':
            continue
        else:
            break

    tl.files.save_npz(net_out.all_params, name='n.npz', sess=sess)
    print('model saved')
    break