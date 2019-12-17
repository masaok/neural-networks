#!/usr/bin/env python3 

import numpy as np
from sklearn import datasets as skdata
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from matplotlib import pyplot as plt
# import matplotlib.gridspec as gridspec


"""
Name: Kitamura, Masao

Collaborators: Doe, Jane

Collaboration details: Discussed <function name> implementation details with Jane Doe.

pip install tensorflow==1.14
pip install tensorflow-gpu==1.14

Tuners:
- learning rate
- optimizers  (Adam, GradientDescent, RMSprop (belief propagation))
- activations  (relu, leaky_relu)
- decay functions  (polynomial, exponential, cosine)
- decay rate (learning_rate start/end)
- preprocessing (standardization, normalization, PCA (dimension reduction))
- network architecture
  - number of neurons (d dimensions, n is number of examples)
  - number of layers
    - Adding more layers is the most expensive thing you can do, so do it last
    - Neurons is tools for your brain to work with (crayons for rainbow)
    - Layers is more knowledge (knowing what a rainbow looks like)

90% for 80+ accuracy plus a good report.
For every percent over 95% will be 1% direct to grade.

Run like this:
watch -d './assignment4.py 2>/dev/null | tail -10'

"""

# n_batch = 32    # Size of each batch
n_batch = 16    # 91%
# n_batch = 8      # 91%
# n_batch = 4       # 91%

# n_epoch = 40    # Number of times to go through training data
n_epoch = 20   # Number of times to go through training data

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
  # Learning rate can change over time
  learning_rate_start = 5e-3  # changing this step size (started at 1e-4)
  learning_rate_end = 1e-20   # Started at 1e-5, but a smaller end is higher accuracy

  # Learning rate decay function (polynomial, exponential, cosine)

  # Polynomial Decay
  # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/polynomial_decay
  learning_rates = tf.train.polynomial_decay(
    learning_rate_start,
    global_step=global_step,
    decay_steps=n_epoch * n_step_train,
    end_learning_rate=learning_rate_end,
    power=0.5
  )

  # Exponential decay is about the same as polynomial decay
  # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/exponential_decay
  # learning_rates = tf.train.exponential_decay(
  #   learning_rate_start,
  #   global_step=global_step,
  #   decay_rate=1,
  #   decay_steps=n_epoch * n_step_train,
  #   # end_learning_rate=learning_rate_end,
  #   # power=0.5
  # )

  # Cosine decay drops accuracy about 5%
  # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/cosine_decay
  # learning_rates = tf.train.cosine_decay(
  #   learning_rate_start,
  #   global_step=global_step,
  #   decay_steps=n_epoch * n_step_train 
  # )


  # TODO: Set up optimizer
  # optimizer = tf.train.GradientDescentOptimizer(learning_rates)
  optimizer = tf.train.AdamOptimizer(learning_rates)

  # optimizer = tf.train.Optimizer(learning_rates)
  # optimizer = tf.train.RMSprop(learning_rates) # fail
  # optimizer = tf.train.SGD(learning_rates) # fail

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

  # 64 x 32 x 10 neurons
  # fc1 = tf.contrib.layers.fully_connected(x_input,
  #   num_outputs=64,
  #   activation_fn=tf.nn.relu  # Or you can use: tf.nn.relu  or tf.nn.relu
  # )

  # fc2 = tf.contrib.layers.fully_connected(fc1,
  #   num_outputs=32,
  #   activation_fn=tf.nn.relu  # Or leaky_relu
  # )

  # outputs = tf.contrib.layers.fully_connected(fc2,
  #   num_outputs=10,
  #   activation_fn=tf.identity
  # )

  # Double neurons
  # fc1 = tf.contrib.layers.fully_connected(x_input,
  #   num_outputs=128,
  #   activation_fn=tf.nn.relu  # Or you can use: tf.nn.relu  or tf.nn.relu
  # )

  # fc2 = tf.contrib.layers.fully_connected(fc1,
  #   num_outputs=64,
  #   activation_fn=tf.nn.relu  # Or leaky_relu
  # )

  # outputs = tf.contrib.layers.fully_connected(fc2,
  #   num_outputs=10,
  #   activation_fn=tf.identity
  # )

  # Extra layer with (10x neurons)
  # fc1 = tf.contrib.layers.fully_connected(x_input,
  #   num_outputs=128,
  #   activation_fn=tf.nn.relu  # Or you can use: tf.nn.relu  or tf.nn.relu
  # )

  # fc2 = tf.contrib.layers.fully_connected(fc1,
  #   num_outputs=64,
  #   activation_fn=tf.nn.relu  # Or leaky_relu
  # )

  # fc3 = tf.contrib.layers.fully_connected(fc2,
  #   num_outputs=32,
  #   activation_fn=tf.nn.relu  # Or leaky_relu
  # )

  # fc4 = tf.contrib.layers.fully_connected(fc3,
  #   num_outputs=16,
  #   activation_fn=tf.nn.relu  # Or leaky_relu
  # )

  num_outputs = 256
  output_counts = []
  output_counts.append(num_outputs)
  fc1 = tf.contrib.layers.fully_connected(x_input,
    num_outputs=num_outputs,
    activation_fn=tf.nn.relu  # Or you can use: tf.nn.relu  or tf.nn.relu
  )
  num_outputs = int(num_outputs / 2)
  layers = 1
  while num_outputs > 10:
    print("num_outputs: " + str(num_outputs))
    output_counts.append(num_outputs)
    fc1 = tf.contrib.layers.fully_connected(fc1,
      num_outputs=num_outputs,
      activation_fn=tf.nn.relu
    )
    num_outputs = int(num_outputs / 2)
    layers += 1

  outputs = tf.contrib.layers.fully_connected(fc1,
    num_outputs=10,
    activation_fn=tf.identity
  )
  layers += 1
  output_counts.append(10)

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

    # print('Epoch={}  Loss={}'.format(epoch, np.mean(np.asarray(losses))))

    # Validate your results
    predicts = np.zeros([n_validate])
    for step in range(n_step_validate):
      # TODO: Iterate over batches in the validation set and make predictions
      batch_start = step * n_batch
      batch_end = batch_start + n_batch 

      # Don't need to supervise the system on validation data
      feed_dict = { x_input : x_validate[batch_start:batch_end, ...] }

      predicts[batch_start:batch_end, ...] = session.run(predictions, feed_dict=feed_dict)

    # Training vs Validation vs Test sets
    # https://stackoverflow.com/questions/2976452/whats-is-the-difference-between-train-validation-and-test-set-in-neural-netwo

    # What is overfitting?
    # https://towardsdatascience.com/deep-learning-overfitting-846bf5b35e24

    # TODO: Evaluate accuracy
    # Compare with y_validate
    score = np.mean(np.where(predicts == y_validate, 1.0, 0.0))
    # print('Validation Accuracy={}'.format(score))
    print('Epoch={}  Loss={}  Accuracy={}'.format(epoch, np.mean(np.asarray(losses)), score))

  # Test your model
  predicts = np.zeros([n_test])
  for step in range(n_step_test):
    # TODO: Iterate over batches in the testing set and make predictions
    batch_start = step * n_batch
    batch_end = batch_start + n_batch 
    feed_dict = { x_input : x_test[batch_start:batch_end, ...]}
    predicts[batch_start:batch_end, ...] = session.run(predictions, feed_dict=feed_dict)


  # # Perceptron accuracy
  # scikit_perceptron = Perceptron(penalty=None, alpha=0.0, tol=1e-5)
  # scikit_perceptron.fit(x_train, y_train)
  # scikit_perceptron_scores = scikit_perceptron.score(x_test, y_test)
  # print('Scikit-learn Perceptron Testing Accuracy={}'.format(scikit_perceptron_scores))

  # # Perceptron accuracy
  # scikit_logistic = LogisticRegression(solver='liblinear')
  # scikit_logistic.fit(x_train, y_train)
  # scikit_logistic_scores = scikit_logistic.score(x_test, y_test)
  # print('Scikit-learn Logistic Regression Testing Accuracy={}'.format(scikit_logistic_scores))

  # Print tuning variables
  print("learning_rate_start: " + str(learning_rate_start))
  print("learning_rate_end: " + str(learning_rate_end))

  print("n_batch: " + str(n_batch))
  print("n_epoch: " + str(n_epoch))
  print("layers: " + str(layers))
  print("neurons: " + str(output_counts))

  # TODO: Evaluate accuracy
  score = np.mean(np.where(predicts == y_test, 1.0, 0.0))
  print('Our Neural Network Testing Accuracy={}'.format(score))

  # TODO: Show 5 by 5 plot of examples from test set predictions
  # with each subplot titled with y=... y_hat=...

  fig = plt.figure()
  # fig = plt.figure(dpi=100)
  plt.subplots_adjust(top=1.2)
  plt.title('Test set predictions', y=-0.06)
  plt.axis('off')
  # plt.margins(10)
  for i in range(25):
    ax = fig.add_subplot(5, 5, i+1)
    ax.set_title('y={} h={}'.format(int(y_test[i]), int(predicts[i])))
    ax.axis('off')
    ax.margins(10)
    ax.imshow(np.reshape(x_test[i, ...], [8, 8]), cmap='gray')

# gs1 = gridspec.GridSpec(1, 1)
# gs1.tight_layout(plt)

plt.show(block=True)
