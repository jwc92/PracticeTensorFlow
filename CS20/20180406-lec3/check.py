import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import utils

DATA_FILE = 'data/birth_life_2010.txt'
data, n_samples = utils.read_birth_life_data(DATA_FILE)
X = tf.placeholder(tf.float32, shape = (), name='X')
Y = tf.placeholder(tf.float32, shape = (), name='Y')

with tf.Session() as sess:
    print(sess.run(tf.shape(X)))

print(np.shape(1))