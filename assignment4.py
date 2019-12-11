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
x = np.asarray(digits.data, np.float32)
y = np.asarray(digits.target, np.uint8)

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

  # TODO: Set up optimizer

  print('Building computational graph...')
  # TODO: Define placeholders for x and y inputs

  # TODO: Convert y into one hot vector form

  # TODO: Create a neural network
  # e.g. 3-layer network with 64, 32, and 10 neurons with ReLU

  # TODO: Apply softmax to outputs and take max

  # TODO: Compute cross entropy with softmax activations

  # TODO: Compute gradients

  # TODO: Apply gradients to update weights

  # Create a Tensorflow session
  session = tf.Session()
  # Initialize all variables
  session.run(tf.global_variables_initializer())
  session.run(tf.local_variables_initializer())

  for epoch in range(1, n_epoch+1):
    # TODO: Shuffle data

    # Create a list to store losses over the epoch
    losses = []
    for step in range(n_step_train):
      # TODO: Iterate over batches in the epoch and train model

      # TODO: Append loss to losses

    print('Epoch={}  Loss={}'.format(epoch, np.mean(np.asarray(losses))))

    # Validate your results
    predicts = np.zeros([n_validate])
    for step in range(n_step_validate):
      # TODO: Iterate over batches in the validation set and make predictions

    # TODO: Evaluate accuracy
    score = 0.0
    print('Validation Accuracy={}'.format(score))

  # Test your model
  predicts = np.zeros([n_test])
  for step in range(n_step_test):
    # TODO: Iterate over batches in the testing set and make predictions

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
