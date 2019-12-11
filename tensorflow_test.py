#!/usr/bin/python

import numpy as np
# from sklearn import datasets as skdata
# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from matplotlib import pyplot as plt

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)

#Session = tf.Session()
Session = tf.compat.v1.Session()
x = tf.ones([1000,1000])
y = 2 * tf.ones([1000,1000])
z = x * y - x
Session.run(z)
