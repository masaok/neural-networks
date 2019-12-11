#!/usr/bin/env python 

import numpy as np
from sklearn import datasets as skdata
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from matplotlib import pyplot as plt


"""
Name: Doe, John (Please write names in <Last Name, First Name> format)

Collaborators: Doe, Jane (Please write names in <Last Name, First Name> format)

Collaboration details: Discussed <function name> implementation details with Jane Doe.
"""

n_batch = 32    # Size of each batch
n_epoch = 40    # Number of times to go through training data

digits = skdata.load_digits()

# Select data
x = np.asarray(digits.data, np.float32)  # (-1, 64)
y = np.asarray(digits.target, np.uint8)  # (-1) -> {0, 1, 2, 3, ..., 9}

# Define data split
n_train = 1440
n_validate = 160
n_test = 180

# Training set
x_train = x[0:n_train, ...]
y_train = y[0:n_train]
n_step_train = n_train//n_batch

# Validation set
x_validate = x[n_train:n_train+n_validate, ...]
y_validate = y[n_train:n_train+n_validate]
n_step_validate = n_validate//n_batch

# Testing set
x_test = x[n_train+n_validate:n_train+n_validate+n_test, ...]
y_test = y[n_train+n_validate:n_train+n_validate+n_test]
n_step_test = n_test//n_batch


with tf.Graph().as_default():
  global_step = tf.Variable(0, trainable=False)

  # TODO: Set up learning rate with polynomial decay
  learning_rate_start = 1e-4
  learning_rate_end = 1e-5

  # learning rate decay function
  learning_rates = tf.train.polynomial_decay(
    learning_rate_start,
    global_step=global_step,
    decay_steps=n_epoch * n_step_train,
    end_learning_rate=learning_rate_end,
    power=0.5
  )

  # TODO: Set up optimizer
  optimizer = tf.train.GradientDescentOptimizer(learning_rates)

  print('Building computational graph...')
  # TODO: Define placeholders for x and y inputs

  # Placeholders are entrypoints to the graph
  x_input = tf.placeholder(tf.float32, shape=[n_batch, x.shape[1]])
  y_input = tf.placeholder(tf.uint8, shape=[n_batch])

  # TODO: Convert y into one hot vector form (a bitmask)
  num_classes = 10
  labels = tf.one_hot(y_input, num_classes, 1.0, 0.0)

  # TODO: Create a neural network
  # e.g. 3-layer network with 64, 32, and 10 neurons with ReLU
  # Transform (Project), reduce, predict (these what the layers represent)

  # Relu has linear activation, for easier gradients that means faster convergence
  # With Sigmoid, you can bound your predictions to 0 and 1
  # With Hyperbolic Tangent, you can bound between -1 and 1
  # With Leaky Relu, you can have negative prediction
  fc1 = tf.contrib.layers.fully_connected(x_input,
    num_outputs=64,
    activation_fn=tf.nn.sigmoid  # Or you can use: tf.nn.relu
  )

  fc2 = tf.contrib.layers.fully_connected(fc1,
    num_outputs=32,
    activation_fn=tf.nn.sigmoid
  )

  outputs = tf.contrib.layers.fully_connected(fc2,
    num_outputs=10,
    activation_fn=tf.identity
  )

  # TODO: Apply softmax to outputs and take max
  predictions = tf.argmax(tf.nn.softmax(outputs), -1)  # -1 is the last dimension

  # TODO: Compute cross entropy with softmax activations
  total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=outputs)

  # TODO: Compute gradients
  gradients = optimizer.compute_gradients(total_loss)

  # TODO: Apply gradients to update weights
  train_op = optimizer.apply_gradients(gradients, global_step=global_step)

  # Create a Tensorflow session
  session = tf.Session()
  # Initialize all variables
  session.run(tf.global_variables_initializer())
  session.run(tf.local_variables_initializer())

  for epoch in range(1, n_epoch+1):
    # TODO: Shuffle data
    order = np.random.permutation(x_train.shape[0])
    x_train_epoch = x_train[order, ...]
    y_train_epoch = y_train[order, ...]

    # Create a list to store losses over the epoch
    losses = []

    # A step is a time step in your training process
    # Grab a chunk of data (x_train_epoch, y_train_epoch)
    # Feed it into the network (graph)
    # Append to losses
    for step in range(n_step_train):
      # TODO: Iterate over batches in the epoch and train model
      batch_start = step * n_batch
      batch_end = batch_start + n_batch
      feed_dict = { x_input : x_train_epoch[batch_start:batch_end, ...],
                    y_input : y_train_epoch[batch_start:batch_end, ...]
      }

      # The blank _ is a value we don't care about
      loss, _ = session.run([total_loss, train_op], feed_dict=feed_dict)

      # TODO: Append loss to losses
      losses.append(loss)

    print('Epoch={}  Loss={}'.format(epoch, np.mean(np.asarray(losses))))

    # Validate your results
    predicts = np.zeros([n_validate])
    for step in range(n_step_validate):
      # TODO: Iterate over batches in the validation set and make predictions
      print("hi")

    # TODO: Evaluate accuracy
    score = 0.0
    print('Validation Accuracy={}'.format(score))

  # Test your model
  predicts = np.zeros([n_test])
  for step in range(n_step_test):
    # TODO: Iterate over batches in the testing set and make predictions
    print("hello")

  # Perceptron accuracy
  scikit_perceptron = Perceptron(penalty=None, alpha=0.0, tol=1e-5)
  scikit_perceptron.fit(x_train, y_train)
  scikit_perceptron_scores = scikit_perceptron.score(x_test, y_test)
  print('Scikit-learn Perceptron Testing Accuracy={}'.format(scikit_perceptron_scores))

  # Perceptron accuracy
  scikit_logistic = LogisticRegression(solver='liblinear')
  scikit_logistic.fit(x_train, y_train)
  scikit_logistic_scores = scikit_logistic.score(x_test, y_test)
  print('Scikit-learn Logistic Regression Testing Accuracy={}'.format(scikit_logistic_scores))

  # TODO: Evaluate accuracy
  score = 0.0
  print('Our Neural Network Testing Accuracy={}'.format(score))

  # TODO: Show 5 by 5 plot of examples from test set predictions
  # with each subplot titled with y=... y_hat=...
