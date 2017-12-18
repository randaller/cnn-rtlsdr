import dataset
import tensorflow as tf
import os
import time
from datetime import timedelta
import math
import random
import numpy as np

batch_size = 16
num_inputs = 2
train_path = 'training_data'

classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
num_classes = len(classes)

data = dataset.read_train_sets(train_path, classes, validation_size=0.3)

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, 96, 128, num_inputs], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 128


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer_tim(input, num_input_channels, conv_filter_size1, conv_filter_size2, num_filters):
    weights = create_weights(shape=[conv_filter_size1, conv_filter_size2, num_input_channels, num_filters])
    biases = create_biases(num_filters)
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases
    layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer = tf.nn.relu(layer)
    # layer = tf.nn.dropout(layer, 0.5)
    return layer


def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases
    layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer = tf.nn.relu(layer)
    return layer


def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer


def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


'''
layer_conv1 = create_convolutional_layer(input=x,
                                         num_input_channels=num_inputs,
                                         conv_filter_size=filter_size_conv1,
                                         num_filters=num_filters_conv1)

layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                         num_input_channels=num_filters_conv1,
                                         conv_filter_size=filter_size_conv2,
                                         num_filters=num_filters_conv2)

layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                         num_input_channels=num_filters_conv2,
                                         conv_filter_size=filter_size_conv3,
                                         num_filters=num_filters_conv3)
'''
# Tim O'Shea network   https://arxiv.org/pdf/1602.04105.pdf
layer_conv1 = create_convolutional_layer_tim(input=x,
                                             num_input_channels=num_inputs,
                                             conv_filter_size1=1,
                                             conv_filter_size2=3,
                                             num_filters=64)

layer_conv3 = create_convolutional_layer_tim(input=layer_conv1,
                                             num_input_channels=64,
                                             conv_filter_size1=2,
                                             conv_filter_size2=3,
                                             num_filters=16)

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat, num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=fc_layer_size, use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1, num_inputs=fc_layer_size, num_outputs=num_classes, use_relu=False)

y_pred = tf.nn.softmax(layer_fc2, name='y_pred')
y_pred_cls = tf.argmax(y_pred, axis=1)
session.run(tf.global_variables_initializer())

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session.run(tf.global_variables_initializer())


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoch {0}, Train Acc: {1:>6.1%}, Val Acc: {2:>6.1%}, Val Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


saver = tf.train.Saver()
# loading pre-trained model to continue training
if (os.path.exists('checkpoint')):
    saver.restore(session, tf.train.latest_checkpoint('./'))

for i in range(0, 25000):

    x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
    x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

    feed_dict_tr = {x: x_batch, y_true: y_true_batch}
    feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}

    session.run(optimizer, feed_dict=feed_dict_tr)

    if i % int(data.train.num_examples / batch_size) == 0:
        val_loss = session.run(cost, feed_dict=feed_dict_val)
        epoch = int(i / int(data.train.num_examples / batch_size))

        show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
        saver.save(session, './rtlsdr-model')
