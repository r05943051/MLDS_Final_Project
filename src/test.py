##MLDS final project##
__author__ = 'CHEN L.'

import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _RNNCell
import numpy as np
import os

data_dir = '../MNIST_data/'
summary_dir = '../summary/'
np_dir = '../numpy_array/'
saving_filenames = ['gradient_images', 'std_images']
training_filenames = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte']
validation_filenames = ['t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']

num_col = 28
num_row = 28
num_px = num_col * num_row
num_classes = 10
num_examples = 60000
num_examples_valiation = 10000

batch_size = 32
lr = 5e-5
nb_epoch = 8
batch_per_epoch = num_examples / batch_size

display_step = 50
sample_gradient_step = 1
validation_step = 900
batch_per_validation = 150

def _linear(inputs, n_hidden, scope=None, reuse=None):
   print inputs
   batch_size, input_dim = inputs.get_shape().as_list()
   with tf.variable_scope(scope or 'linear', reuse=reuse) as scope:
      weights = tf.get_variable('weights', shape=[input_dim, n_hidden])
      biases = tf.get_variable('biases', shape=[n_hidden], initializer=tf.zeros_initializer())
   return tf.nn.bias_add(tf.matmul(inputs, weights), biases)

class LSTMCell(_RNNCell):
   def __init__(self, num_units, forget_bias=1.0, 
                gate_activation=None, output_activation=None, reuse=None):
      super(LSTMCell, self).__init__()
      self._num_units = num_units
      self._forget_bias = forget_bias
      self._gate_activation = gate_activation or tf.tanh
      self._output_activation = output_activation or tf.tanh
      self._reuse = reuse
      self._cell_grad = []
   @property
   def state_size(self):
      return tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)
   @property
   def output_size(self):
      return self._num_units
   @property
   def cell_grad(self):
      return self._cell_grad
   def __call__(self, inputs, state):
      print "aa"
      sigmoid = tf.sigmoid
      c, h = state
      with tf.variable_scope('lstm', reuse=self._reuse) as scope:
         inp = tf.concat([inputs, h], -1)
         lstm_matrix = _linear(inp, 4 * self._num_units, scope='lstm_linear', reuse=self._reuse)
         i, j, f, o = tf.split(value=lstm_matrix, num_or_size_splits=4, axis=1)
         c = (sigmoid(f + self._forget_bias) * c + sigmoid(i) * self._gate_activation(j))
         print c, state.c
         self._cell_grad.append(tf.gradients(c, state.c)[0])
         h = sigmoid(o) * self._output_activation(c)
      return h, tf.contrib.rnn.LSTMStateTuple(c, h)

def load_data(filenames, flatten, validation):
   if validation:
      _num_examples = num_examples_valiation
   else:
      _num_examples = num_examples
   print "Loading in training datas...."
   images = open(os.path.join(data_dir, filenames[0]), 'rb')
   labels = open(os.path.join(data_dir, filenames[1]), 'rb')
   images.seek(16)
   labels.seek(8)
   ibyte = images.read(1)
   lbyte = labels.read(1)
   x_train = np.zeros((_num_examples, num_px))
   y_train = np.zeros(_num_examples)
   pos_x = 0 
   while lbyte:
      for pos_y in range(num_px):
         x_train[pos_x][pos_y] = ord(ibyte)
         ibyte = images.read(1)
      y_train[pos_x] = ord(lbyte)
      lbyte = labels.read(1)
      pos_x += 1
   x_train /= 255.0
   if not flatten:
      x_train = np.reshape(x_train, [_num_examples, num_col, num_row])
   else:
      x_train = np.reshape(x_train, [_num_examples, num_px, 1])
   images.close()
   labels.close()
   return x_train, y_train

def data_generator(batch_size, flatten, training=True):
   p = 0
   if training:
      x_train, y_train = load_data(training_filenames, flatten, False)
      _num_examples = num_examples
   else:
      x_train, y_train = load_data(validation_filenames, flatten, True)
      _num_examples = num_examples_valiation
   while 1:
      if p + batch_size >= _num_examples:
         tmp_x = x_train[p: ]
         tmp_y = y_train[p: ]
         seed = np.arange(_num_examples)
         np.random.shuffle(seed)
         x_train = x_train[seed]
         y_train = y_train[seed]
         x = np.concatenate([tmp_x, x_train[: p + batch_size - _num_examples]], axis=0)
         y = np.concatenate([tmp_y, y_train[: p + batch_size - _num_examples]], axis=0)
         yield x, y
         p = p + batch_size - _num_examples
      yield x_train[p: p + batch_size], y_train[p: p + batch_size]
      p += batch_size

def static_LSTM(inputs, n_hidden, gate_activation, output_activation, seqlen=None, scope=None, reuse=None):
   print inputs
   batch_size, time_step, embed_dim = inputs.get_shape().as_list()
   inputs = tf.split(inputs, time_step, axis=1)
   inputs = [tf.reshape(x, [batch_size, embed_dim]) for x in inputs]
   with tf.variable_scope(scope or "dynamic_LSTM", reuse=reuse):
      lstm_cell = LSTMCell(n_hidden, gate_activation=gate_activation, output_activation=output_activation)
      outputs, _ = tf.contrib.rnn.static_rnn(lstm_cell, inputs, sequence_length=seqlen, dtype=tf.float32)
      outputs = tf.stack(outputs, axis=1)
      return outputs, lstm_cell.cell_grad

def multiLSTM(inputs, n_hiddens, gate_activation, output_activation, keep_prob, scope=None, reuse=None):
   ret_list = []
   with tf.variable_scope(scope or "multi_LSTM", reuse=reuse):
      rec_outputs = inputs
      for idx, n_hidden in enumerate(n_hiddens):
         scope_name = 'LSTMlayer_' + str(idx)
         if idx % 2:
            rec_outputs = tf.nn.dropout(rec_outputs, keep_prob)
         rec_outputs, z = static_LSTM(rec_outputs, n_hidden, gate_activation, output_activation, scope=scope_name, reuse=reuse)
         ret_list.append(rec_outputs)
      return ret_list
   
def leaky_relu(inputs):
   alpha = 0.5
   return tf.maximum(alpha * inputs, inputs)

def process_grad(labels, gradients, buf, total_buf, total_std):
   def gen_buf():
      buf = []
      for i in xrange(num_classes):
         buf.append(np.empty([0, num_col, num_row]))
      return buf
   if buf is None:
      buf = gen_buf()
   if total_buf is None:
      total_buf = gen_buf()
   if total_std is None:
      total_std = gen_buf()
   for label, gradient in zip(labels, gradients):
      gradient = np.expand_dims(gradient, 0)
      buf[label] = np.concatenate([buf[label], gradient], axis=0)
   sample_size = sample_gradient_step * batch_size
   for idx, grad in enumerate(buf):
      if grad.shape[0] >= sample_size:
         tmp = np.mean(grad[:sample_size], axis=0, keepdims=True)
         std = np.sqrt(np.mean((grad - np.squeeze(tmp)) ** 2, axis=0, keepdims=True))
         total_buf[idx] = np.concatenate([total_buf[idx], tmp], axis=0)
         total_std[idx] = np.concatenate([total_std[idx], std], axis=0)
         buf[idx] = grad[sample_size:]
   return buf, total_buf, total_std

def train(training_type, gate_activation, output_activation):
   activation_map = {'tanh': tf.tanh,
                     'sigmoid': tf.sigmoid,
                     'relu': tf.nn.relu,
                     'elu': tf.nn.elu,
                     'leaky_relu': leaky_relu}
   assert gate_activation in activation_map
   assert output_activation in activation_map
   assert training_type in ['LTSL', 'STDL', 'LTDL']
   g_act = activation_map[gate_activation]
   o_act = activation_map[output_activation]
   if training_type == 'LTSL':
      flatten = True
      _shape = [batch_size, num_px, 1]
      n_hiddens = [32]
   elif training_type == 'STDL':
      flatten = False
      _shape = [batch_size, num_col, num_row]
      n_hiddens = [64, 32, 32, 32, 32, 32, 32, 32, 16, 8]
   elif training_type == 'LTDL':
      flatten = True
      _shape = [batch_size, num_px, 1]
      n_hiddens = [64, 32, 16, 8]

   data_gen = data_generator(batch_size, flatten)
   val_data_gen = data_generator(batch_size, flatten, False)
   images = tf.placeholder(dtype=tf.float32, shape=_shape)
   labels = tf.placeholder(dtype=tf.int32)
   keep_prob = tf.placeholder(dtype=tf.float32)
   place_holders = [images, labels]
   all_layer_outputs = multiLSTM(images, n_hiddens, g_act, o_act, keep_prob)
   outputs = all_layer_outputs[-1]
   output = outputs[:, -1]
   output = _linear(output, num_classes, scope='Decision')
   output = tf.reshape(output, [batch_size, num_classes])
   loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output)
   loss = tf.reduce_mean(loss)
   pred = tf.cast(tf.argmax(output, axis=-1), tf.int32)
   accuracy = tf.cast(tf.equal(pred, labels), tf.float32)
   accuracy = tf.reduce_mean(accuracy)
   train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

   _name = training_type + '_Gate_Activation_' + gate_activation + '_Output_activation_' + output_activation
   with tf.name_scope(_name):
      grad_through_time = tf.abs(tf.gradients(loss, images)[0])
      grad_through_time = tf.reshape(grad_through_time, [batch_size, num_col, num_row])
      grad_through_layer = tf.gradients(loss, all_layer_outputs)
      grad_through_layer = [tf.reduce_mean(grad_layer ** 2) for grad_layer in grad_through_layer]

   gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
   with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      train_writer = tf.summary.FileWriter(summary_dir + 'train', sess.graph)
      init = tf.global_variables_initializer()
      sess.run(init)
      gradient_buffer, total_buffer, total_std_buffer = None, None, None
      for epoch in xrange(nb_epoch):
         _total_loss, _total_acc, = 0, 0
         for batch in xrange(batch_per_epoch):
            input_datas = next(data_gen)
            _labels = input_datas[1].astype(np.int32)
            feed_dict = {a: b for a, b in zip(place_holders, input_datas)}
            feed_dict[keep_prob] = 0.8
            run_list = [train_op, loss, accuracy, grad_through_time]
            _, _loss, _acc, _grad = sess.run(run_list, feed_dict=feed_dict)
            _total_loss += _loss
            _total_acc += _acc
            gradient_buffer, total_buffer, total_std_buffer = process_grad(_labels, _grad, gradient_buffer, total_buffer, total_std_buffer)
            if (batch + 1) % display_step == 0:
               print "Epoch : ", epoch + 1, "Batch : ", batch + 1, '/', batch_per_epoch, \
                     "Type : ", _name, \
                     "Loss : ", "{:.6f}".format(_total_loss / display_step), \
                     "Accuracy : ", "{:.6f}".format(_total_acc / display_step)
               _total_loss, _total_acc = 0, 0
            if (batch + 1) % validation_step == 0:
               _total_loss_val, _total_acc_val = 0, 0
               for val_batch in xrange(batch_per_validation):
                  feed_dict = {a: b for a, b in zip(place_holders, next(val_data_gen))}
                  feed_dict[keep_prob] = 1.0
                  _loss, _acc = sess.run([loss, accuracy], feed_dict=feed_dict)
                  _total_loss_val += _loss
                  _total_acc_val += _acc
               print "Validation, ", \
                     "Type : ", _name, \
                     "Loss : ", "{:.6f}".format(_total_loss_val / batch_per_validation), \
                     "Accuracy : ", "{:.6f}".format(_total_acc_val / batch_per_validation)
   save_dict = {str(idx): x for idx, x in enumerate(total_buffer)}
   save_dict_std = {str(idx): x for idx, x in enumerate(total_std_buffer)}
   with open(os.path.join(np_dir, saving_filenames[0]), 'wb') as f:
      np.savez(f, **save_dict)
   with open(os.path.join(np_dir, saving_filenames[1]), 'wb') as f:
      np.savez(f, **save_dict_std)

train('STDL', 'elu', 'elu')
