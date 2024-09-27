import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from functions.py import *

# global parameters

size = 784
n_classes = 10

# the dataset is loaded and normalized

(x_train, y_train_labels), (x_test, y_test_labels) = datasets.mnist.load_data()

x_train = x_train.reshape((-1,784))/255
x_test = x_test.reshape((-1,784))/255

y_train = tf.one_hot(y_train_labels, 10) @ np.transpose(att)
y_test = tf.one_hot(y_test_labels, 10) @ np.transpose(att)

# attractors are created

beta = np.sqrt(1/size)
gamma = 1/(8*size)

xp = (beta + np.sqrt(beta**2 - 4* gamma))/2
rp = non_linearity(xp, gamma=gamma).numpy()

att = np.ones((size,n_classes)) * rp

for c in range(n_classes):
    att[c*78 : (c+1)*78, c] = 0


mod = CVFR_model(size, att,
                         deltat=0.03,
                         max_steps=300,
                         sigma=0.1,
                         n_classes = n_classes)
mod.compile(optimizer=tf.optimizers.Adam(learning_rate=0.005),
                loss='mse')
mod.fit(x_train, y_train , epochs=100, batch_size=500, validation_split=0.1)
