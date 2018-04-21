#!/usr/bin/env python3


import tensorflow as tf
import numpy as np
import random
import time
import os
import cv2
import logging
import json

from tensorflow.python.training.moving_averages import assign_moving_average
from tensorflow.python.client import timeline

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


BATCH_SIZE = 64
IMAGE_SIZE = 512
NUM_CHANNEL = 1
NUM_LABELS = 2
LEARNING_RATE = 0.001

NUM_STEP = 120000
STEP_CLAC_STAT = 150

TRAIN_DATA_LIST = 'train_data_list.txt'
TEST_DATA_LIST = 'test_data_list.txt'
LOG_PATH = '{}/train_log'.format(__file__[:-3])
PARAM_PATH = '{}/params'.format(__file__[:-3])


def batch_norm(x, phase, eps=1e-05, decay=0.9, name='batch_norm'):
  with tf.variable_scope(name):
    params_shape = x.shape[-1:]

    pop_mean = tf.get_variable('pop_mean', params_shape, initializer=tf.zeros_initializer, trainable=False)
    pop_variance = tf.get_variable('pop_variance', params_shape, initializer=tf.zeros_initializer, trainable=False)

    beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
    gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)

    if phase == 'training':
      mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

      return tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)

    elif phase == 'calc_stat':
      mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

      update_pop_mean = tf.assign(pop_mean, pop_mean + (mean / STEP_CLAC_STAT))
      update_pop_variance = tf.assign(pop_variance, pop_variance + (variance / STEP_CLAC_STAT))

      with tf.control_dependencies([update_pop_mean, update_pop_variance]):
        return tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)

    elif phase == 'evaluation':
      return tf.nn.batch_normalization(x, pop_mean, pop_variance, beta, gamma, eps)


def model(x, phase, scope='class', reuse=None):
  with tf.variable_scope(scope, reuse=reuse):

    # x = tf.Print(x, [x], 'x: ')

    hpf = np.zeros([5, 5, 1, 1], dtype=np.float32)
    hpf[:, :, 0, 0] = np.array(
      [[-1, 2, -2, 2, -1],
      [2, -6, 8, -6, 2],
      [-2, 8, -12, 8, -2],
      [2, -6, 8, -6, 2],
      [-1, 2, -2, 2, -1]], dtype=np.float32) / 12


    kernel0 = tf.Variable(hpf,name='kernel0',trainable=False)
    output = tf.nn.conv2d(x, filter=kernel0, strides=[1, 1, 1, 1], padding='SAME', name='conv0')


    with tf.variable_scope('group1'):
     kernel1 = tf.get_variable('kernel1', shape=[5, 5, 1, 8], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
     output = tf.nn.conv2d(output, filter=kernel1, strides=[1, 1, 1, 1], padding='SAME', name='conv1')
     output = tf.abs(output, name='abs1')

     output = batch_norm(output, phase=phase, eps=1e-3, decay=0.6)

     output = tf.nn.tanh(output, name='tanh1')
     output = tf.nn.avg_pool(output, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')


    with tf.variable_scope('group2'):
     kernel2 = tf.get_variable('kernel2', shape=[5, 5, 8, 16], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
     output = tf.nn.conv2d(output, filter=kernel2, strides=[1, 1, 1, 1], padding='SAME', name='conv2')

     output = batch_norm(output, phase=phase, eps=1e-3, decay=0.6)

     output = tf.nn.tanh(output, name='tanh2')
     output = tf.nn.avg_pool(output, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')


    with tf.variable_scope('group3'):
     kernel3 = tf.get_variable('kernel3', shape=[1, 1, 16, 32], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
     output = tf.nn.conv2d(output, filter=kernel3, strides=[1, 1, 1, 1], padding='SAME', name='conv3')

     output = batch_norm(output, phase=phase, eps=1e-3, decay=0.6)

     output = tf.nn.relu(output, name='bn3')
     output = tf.nn.avg_pool(output, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')


    with tf.variable_scope('group4'):
     kernel4 = tf.get_variable('kernel4', shape=[1, 1, 32, 64], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
     output = tf.nn.conv2d(output, filter=kernel4, strides=[1, 1, 1, 1], padding='SAME', name='conv4')

     output = batch_norm(output, phase=phase, eps=1e-3, decay=0.6)

     output = tf.nn.relu(output, name='relu4')
     output = tf.nn.avg_pool(output, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='SAME',name='pool4')


    with tf.variable_scope('group5'):
     kernel5 = tf.get_variable('kernel5', shape=[1, 1, 64, 128], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
     output = tf.nn.conv2d(output, filter=kernel5, strides=[1, 1, 1, 1], padding='SAME', name='conv5')

     output = batch_norm(output, phase=phase, eps=1e-3, decay=0.6)

     output = tf.nn.relu(output, name='relu5')
     output = tf.nn.avg_pool(output, ksize=[1, 32, 32, 1], strides=[1, 1, 1, 1], padding='VALID',name='pool5')


    with tf.variable_scope('group6'):
     output = tf.reshape(output, [-1, 128 * 1 * 1])
     weights = tf.get_variable('weights', [128, 2], initializer=tf.contrib.layers.xavier_initializer())
     bias = tf.get_variable('bias', shape=[2], initializer=tf.constant_initializer(0.0))

     y_ = tf.matmul(output, weights) + bias


    return y_



def optimize(y, y_):
  global_step_op = tf.Variable(0, name='global_step', trainable=False)
  learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step_op, 5000, 0.9, staircase=True)

  loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_, labels=y))
  train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss_op, global_step=global_step_op)

  return train_op, loss_op, global_step_op



def read_labeled_image_list(data_list):
  f = open(data_list, 'r')
  images = []
  labels = []
  for line in f:
    image, label = line.strip("\n").split(' ')
    # print 'list'
    # print image
    # print label
    images.append( image)
    labels.append( int(label))


  return images, labels



def _read_py_function(filename, label):

  # maybe a bug
  filename = filename.decode('utf-8')


  image_decoded = cv2.imread(filename, -1)

  return image_decoded, label


def preprocess(image, label):
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.expand_dims(image, -1)

  return image, label



def get_batch_data(sess, batch_size):
  num_parallel_calls = 4
  num_prefetch = 1

  train_filenames, train_labels = read_labeled_image_list(TRAIN_DATA_LIST)
  train_data = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))

  train_data = train_data.repeat()

  train_data = train_data.map(lambda filename, label: tuple(tf.py_func(
    _read_py_function, [filename, label], [tf.uint8, label.dtype])), num_parallel_calls=num_parallel_calls)

  train_data = train_data.map(preprocess, num_parallel_calls=num_parallel_calls)

  train_data = train_data.batch(batch_size)
  train_data = train_data.prefetch(num_prefetch)


  # -----

  test_filenames, test_labels = read_labeled_image_list(TEST_DATA_LIST)
  test_data = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))

  test_data = test_data.map(lambda filename, label: tuple(tf.py_func(
    _read_py_function, [filename, label], [tf.uint8, label.dtype])), num_parallel_calls=num_parallel_calls)

  test_data = test_data.map(preprocess, num_parallel_calls=num_parallel_calls)

  test_data = test_data.batch(batch_size)
  test_data = test_data.prefetch(num_prefetch)

  # -----

  handle_placeholder = tf.placeholder(tf.string, shape=[])
  iterator = tf.data.Iterator.from_string_handle(handle_placeholder, train_data.output_types, train_data.output_shapes)
  data_batch = iterator.get_next()

  train_iterator = train_data.make_one_shot_iterator()
  test_iterator = test_data.make_initializable_iterator()

  train_handle = sess.run(train_iterator.string_handle())
  test_handle = sess.run(test_iterator.string_handle())

  return data_batch, handle_placeholder, train_handle, test_handle, test_iterator


def set_logger(log_path, mode='a'):
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  if not logger.handlers:
    # Logging to a file
    file_handler = logging.FileHandler(log_path, mode=mode)
    file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)


class TimeLiner:
    _timeline_dict = None

    def update_timeline(self, step_stats):
        fetched_timeline = timeline.Timeline(step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()

        # convert chrome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)


def get_all_variables_with_name(var_name):
    name = var_name + ':0'
    return [var for var in tf.global_variables() if var.name.endswith(name)]

def main():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  sess = tf.Session(config=config)

  data_batch, handle_placeholder, train_handle, test_handle, test_iterator = get_batch_data(sess, BATCH_SIZE)
  x, y = data_batch

  # print(x.dtype, y.dtype)
  # return

  y_train = model(x, phase='training')
  y_calc_stat = model(x, phase='calc_stat', reuse=True)

  train_op, loss_op, global_step_op = optimize(y, y_train)

  y_eval = model(x, phase='evaluation', reuse=True)

  with tf.name_scope("streaming"):
    accuracy, acc_update_op = tf.metrics.accuracy(y, tf.argmax(y_eval, axis=1))

  test_fetches = {
    'accuracy': accuracy,
    'acc_op': acc_update_op
  }


  variable_init = tf.global_variables_initializer()
  sess.run(variable_init)

  bn_stat_var = get_all_variables_with_name('pop_mean')  + get_all_variables_with_name('pop_variance')
  set_bn_stat_zero = tf.initializers.variables(bn_stat_var)

  saver = tf.train.Saver()

  if not os.path.exists(__file__[:-3]):
    os.makedirs(__file__[:-3])

  set_logger(LOG_PATH, mode='w')

  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()
  # many_runs_timeline = TimeLiner()
  # timeline_path = '{}/timeline_merged.json'.format(__file__[:-3])

  for step in range(1, NUM_STEP + 1):
  # for step in range(1, 10 + 1):
    _, loss, global_step = sess.run([train_op, loss_op, global_step_op], feed_dict={handle_placeholder: train_handle}, options=options, run_metadata=run_metadata)

    if step % 100 == 0:
      logging.info('Step: {:d}, loss: {:8f}'.format(global_step, loss))


    if step % 2000 == 0:
      _ = sess.run(set_bn_stat_zero)
      for i in range(STEP_CLAC_STAT):
        _ = sess.run(y_calc_stat, feed_dict={handle_placeholder: train_handle})

      sess.run(test_iterator.initializer)

      with tf.name_scope("streaming"):
        sess.run(tf.local_variables_initializer())

      while True:
        try:
          outputs = sess.run(test_fetches, feed_dict={handle_placeholder: test_handle})
        except tf.errors.OutOfRangeError:
          break

      saver.save(sess, PARAM_PATH)

      logging.info('Accuracy: {:8f}'.format(outputs['accuracy']))




    # many_runs_timeline.update_timeline(run_metadata.step_stats)


  # many_runs_timeline.save(timeline_path)
  # logging.info('Timeline saved to: {}'.format(timeline_path))



if __name__ == '__main__':
  main()




# bn, accuracy, get_data, adversial
