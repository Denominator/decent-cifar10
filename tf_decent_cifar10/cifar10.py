# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from inception.slim import scopes
# from inception.slim import variables

import gzip
import os
import re
import sys
import tarfile
import math

from six.moves import urllib
import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.training import learning_rate_decay
print(learning_rate_decay.__file__)

# from tensorflow.models.image.cifar10_top_model import cifar10_input
import cifar10_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('experiment_prefix', 'experiments/no_momentum_lr_0_1', "Prefix for the current experiment")
# tf.app.flags.DEFINE_string('experiment_prefix', 'exp_wd_0', "Prefix for the current experiment")

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# original settings
# # Constants describing the training process.
# MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
# NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
# LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
# INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# settings from a reference model
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
# INITIAL_LEARNING_RATE = 10.0       # Initial learning rate.
# INITIAL_LEARNING_RATE = 1.0       # Initial learning rate.
# INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.

MOMENTUM = 0.9

# lr dividing schedule (in steps) borrowe directly form the Torch version of the model
# divide by 2 every 25 epochs
# LR_DROP_EVERY_NO_EPOCHS = 25
LR_DROP_EVERY_NO_EPOCHS = 25
LR_DROP_SCALE = 2

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def _variable_with_weight_decay_init_normal(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.random_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def _variable_with_weight_decay_init_xavier(name, shape, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.contrib.layers.xavier_initializer())
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def _variable_with_weight_decay_init_ref_model_linear_uniform(name, shape, half_range, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """

  var = _variable_on_cpu(name, shape,
                    tf.random_uniform_initializer(-half_range, half_range))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  return cifar10_input.distorted_inputs(data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  return cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir,
                              batch_size=FLAGS.batch_size)



# Used to keep the update ops done by batch_norm.
UPDATE_OPS_COLLECTION = '_update_ops_'


# @scopes.add_arg_scope #TODO: figure out why it doesn't work and what did it do
def batch_norm(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               moving_vars='moving_vars',
               activation=None,
               is_training=True,
               trainable=True,
               # restore=True,
               scope=None,
               reuse=None):
  """Adds a Batch Normalization layer.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels]
            or [batch_size, channels].
    decay: decay for the moving average.
    center: If True, subtract beta. If False, beta is not created and ignored.
    scale: If True, multiply by gamma. If False, gamma is
      not used. When the next layer is linear (also e.g. ReLU), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: small float added to variance to avoid dividing by zero.
    moving_vars: collection to store the moving_mean and moving_variance.
    activation: activation function.
    is_training: whether or not the model is in training mode.
    trainable: whether or not the variables should be trainable or not.
    restore: whether or not the variables should be marked for restore.
    scope: Optional scope for variable_op_scope.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
  Returns:
    a tensor representing the output of the operation.
  """

  print("TODO: BN set scale to False, check if it is OK")

  inputs_shape = inputs.get_shape()
  with tf.variable_op_scope([inputs], scope, 'BatchNorm', reuse=reuse):
    axis = list(range(len(inputs_shape) - 1))
    params_shape = inputs_shape[-1:]
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
    # with tf.device('/cpu:0'):
      beta = tf.get_variable('beta',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=trainable,
                                # restore=restore
                                )
    if scale:
    # with tf.device('/cpu:0'):
      gamma = tf.get_variable('gamma',
                                 params_shape,
                                 initializer=tf.ones_initializer,
                                 trainable=trainable,
                                 # restore=restore
                                 )
    # Create moving_mean and moving_variance add them to
    # GraphKeys.MOVING_AVERAGE_VARIABLES collections.
    moving_collections = [moving_vars, tf.GraphKeys.MOVING_AVERAGE_VARIABLES]
    # with tf.device('/cpu:0'):
    moving_mean = tf.get_variable('moving_mean',
                                     params_shape,
                                     initializer=tf.zeros_initializer,
                                     trainable=False,
                                     # trainable=True,
                                     # restore=False,
                                     collections=moving_collections)

    # # debug
    # print("trainable variables")
    # for var in tf.trainable_variables():
    #     print(var.name)

    with tf.device('/cpu:0'):
      moving_variance = tf.get_variable('moving_variance',
                                           params_shape,
                                           initializer=tf.ones_initializer,
                                           trainable=False,
                                           # trainable=True,
                                           # restore=restore,
                                           collections=moving_collections)
    if is_training:
      # Calculate the moments based on the individual batch.
      mean, variance = tf.nn.moments(inputs, axis)

      update_moving_mean = moving_averages.assign_moving_average(
          moving_mean, mean, decay)
      tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
      update_moving_variance = moving_averages.assign_moving_average(
          moving_variance, variance, decay)
      tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
    else:
      # Just use the moving_mean and moving_variance.
      mean = moving_mean
      variance = moving_variance
    # Normalize the activations.
    outputs = tf.nn.batch_normalization(
        inputs, mean, variance, beta, gamma, epsilon)
    outputs.set_shape(inputs.get_shape())
    if activation:
      outputs = activation(outputs)
    return outputs


def create_conv_bn_relu_block(idx, input, no_in_channels, no_out_channels, padding='SAME'):

  # spatial convolution
  with tf.variable_scope('conv' + idx) as scope:
    # kernel = _variable_with_weight_decay('weights', shape=[3, 3, no_in_channels, no_out_channels],
                                         # stddev=1e-4, wd=0.0)

    print("TODO: try using no_in_channels instead of no_out_channels for init of the convnet, MRS init seems to be closer to that")
    n_factor = no_out_channels * 3 * 3 # v.kW*v.kH*v.nOutputPlane from Torch
    init_stddev = math.sqrt(2/n_factor)
    kernel = _variable_with_weight_decay_init_normal('weights', shape=[3, 3, no_in_channels, no_out_channels], stddev=init_stddev, wd=0.0005)

    conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding=padding)
    # print(conv)
    biases = _variable_on_cpu('biases', [no_out_channels], tf.constant_initializer(0.0))
    conv = tf.nn.bias_add(conv, biases)

    # add batch norm here
    conv = batch_norm(conv)

    conv = tf.nn.relu(conv)
    _activation_summary(conv)

  return conv
  # return tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding=padding)

def inference(images):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  # with tf.variable_scope('conv1') as scope:
    # kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                         # stddev=1e-4, wd=0.0)
    # conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    # biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    # bias = tf.nn.bias_add(conv, biases)
    # conv1 = tf.nn.relu(bias, name=scope.name)
    # _activation_summary(conv1)

  print("------------------- building cifar 10 model --------------------")

  ##################

  conv1 = create_conv_bn_relu_block("1", images, 3, 64)
  conv1_dropout = tf.nn.dropout(conv1, 0.7) #TODO: triple check this, so weird that in torch it is 0.3

  # # norm1
  # norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
  #                   name='norm1')

  # # conv2
  # with tf.variable_scope('conv2') as scope:
  #   kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
  #                                        stddev=1e-4, wd=0.0)
  #   conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
  #   biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
  #   bias = tf.nn.bias_add(conv, biases)
  #   conv2 = tf.nn.relu(bias, name=scope.name)
  #   _activation_summary(conv2)

  conv2 = create_conv_bn_relu_block("2", conv1_dropout, 64, 64)

  # pool1
  pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')

  # ##################

  conv3 = create_conv_bn_relu_block("3", pool1, 64, 128)
  conv3_dropout = tf.nn.dropout(conv3, 0.6) #TODO: triple check this, so weird that in torch it is 0.3

  conv4 = create_conv_bn_relu_block("4", conv3_dropout, 128, 128)

  # pool2
  pool2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2') 

  # ##################
  conv5 = create_conv_bn_relu_block("5", pool2, 128, 256)
  conv5_dropout = tf.nn.dropout(conv5, 0.6) #TODO: triple check this, so weird that in torch it is 0.3

  conv6 = create_conv_bn_relu_block("6", conv5_dropout, 256, 256)
  conv6_dropout = tf.nn.dropout(conv6, 0.6) #TODO: triple check this, so weird that in torch it is 0.3

  conv7 = create_conv_bn_relu_block("7", conv6_dropout, 256, 256)

  # pool3
  pool3 = tf.nn.max_pool(conv7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool3') 

  # #################

  conv8 = create_conv_bn_relu_block("8", pool3, 256, 512)
  conv8 = tf.nn.dropout(conv8, 0.6) #TODO: triple check this, so weird that in torch it is 0.3

  conv9 = create_conv_bn_relu_block("9", conv8, 512, 512)
  conv9 = tf.nn.dropout(conv9, 0.6) #TODO: triple check this, so weird that in torch it is 0.3

  conv10 = create_conv_bn_relu_block("10", conv9, 512, 512)

  # pool4
  pool4 = tf.nn.max_pool(conv10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool4') 


  #################
  # TODO: ADD LAST LAYERS, THERE WAS A SIZE PROBLEM

  # padding code, tensorflow has a check on the size of an input vs size of a filter
  paddings = [[0, 0], [1, 1], [1, 1], [0, 0]] # this is just 1 pixel padding on each side for width and height
  conv11 = tf.pad(pool4, paddings, mode='CONSTANT')

  conv11 = create_conv_bn_relu_block("11", conv11, 512, 512, padding='VALID')
  conv11 = tf.nn.dropout(conv11, 0.6) #TODO: triple check this, so weird that in torch it is 0.3

  # padding code, tensorflow has a check on the size of an input vs size of a filter
  paddings = [[0, 0], [1, 1], [1, 1], [0, 0]] # this is just 1 pixel padding on each side for width and height
  conv11 = tf.pad(conv11, paddings, mode='CONSTANT', name="pad_2")

  conv12 = create_conv_bn_relu_block("12", conv11, 512, 512, padding='VALID')
  conv12 = tf.nn.dropout(conv12, 0.6) #TODO: triple check this, so weird that in torch it is 0.3

  # padding code, tensorflow has a check on the size of an input vs size of a filter
  paddings = [[0, 0], [1, 1], [1, 1], [0, 0]] # this is just 1 pixel padding on each side for width and height
  conv12 = tf.pad(conv12, paddings, mode='CONSTANT', name="pad_3")

  conv13 = create_conv_bn_relu_block("13", conv12, 512, 512, padding='VALID')

  # pool5
  pool5 = tf.nn.max_pool(conv13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool5') 

  #### classifier ####

  # DROPOUT
  pool5 = tf.nn.dropout(pool5, 0.5)

  # LINEAR 512 x 512
  # local3, linear ? to 512
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool5, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value

    #debug
    print("reshape dim (pool5 size): " + str(dim))

     #stdv = 1./math.sqrt(self.weight:size(2))
    half_range = 1.0 / math.sqrt(512)

    # weights = _variable_with_weight_decay_init_xavier('weights', shape=[dim, 512], wd=0.0)
    weights = _variable_with_weight_decay_init_ref_model_linear_uniform('weights', shape=[dim, 512], half_range=half_range, wd=0.0005)

    # tf.random_uniform_initializer(-half_range, half_range))
    
    # biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
    # biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
    biases = _variable_on_cpu('biases', [512], tf.random_uniform_initializer(-half_range, half_range))

    local3 = tf.matmul(reshape, weights) + biases
    _activation_summary(local3)
  
  ##################

  #BN
  local3 = batch_norm(local3)

  #ReLU
  local3 = tf.nn.relu(local3, name=scope.name)

  #Dropout
  local3 = tf.nn.dropout(local3, 0.5) #TODO: so weird that in torch it is opposite

  # LINEAR 512 x 10
  # softmax_linear
  with tf.variable_scope('softmax_linear') as scope:

    #stdv = 1./math.sqrt(self.weight:size(2))
    half_range = 1.0 / math.sqrt(512)

    weights = _variable_with_weight_decay_init_ref_model_linear_uniform('weights', shape=[512, NUM_CLASSES], half_range=half_range, wd=0.0005)
    
    # biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.random_uniform_initializer(-half_range, half_range))

    softmax_linear = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Drop learning rate every "epoch_step" epochs
  #if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end

  # Decay the learning rate exponentially based on the number of steps.
  
  # debug
  print("global_step: " + str(global_step))
  print("NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN: " + str(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN))
  print("batch size: " + str(FLAGS.batch_size))
  # epochs_done = math.floor(float(global_step) * FLAGS.batch_size / NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
  # print("epochs done: " + str(epochs_done))

  # # lr decay
  # lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
  #                                 global_step,
  #                                 decay_steps,
  #                                 LEARNING_RATE_DECAY_FACTOR,
  #                                 staircase=True)

  # lr manual control
  lr_boundaries = list()
  lr_values = list()
  for drop_no in range(1, 21):
    LR_DROP_EVERY_NO_STEPS = LR_DROP_EVERY_NO_EPOCHS * num_batches_per_epoch
    lr_boundary = int(drop_no * LR_DROP_EVERY_NO_STEPS)
    lr_boundaries.append(lr_boundary)

    lr_value = INITIAL_LEARNING_RATE / 2 ** (drop_no - 1)
    lr_values.append(lr_value)
  
  print(lr_boundaries)
  print(lr_values)

  # boundaries = [100000, 110000]
  # values = [1.0, 0.5, 0.1]
  lr = learning_rate_decay.piecewise_constant(global_step, lr_boundaries, lr_values)

  # sess = tf.Session()
  # lr_val = sess.run(lr)
  # print("lr_val: " + lr_val)

  # if epochs_done % LR_DROP_EVER_NO_EPOCHS == 0:
  #   lr = lr  / LR_DROP_SCALE


  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    # opt = tf.train.GradientDescentOptimizer(lr)
    opt = tf.train.MomentumOptimizer(lr, MOMENTUM)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
