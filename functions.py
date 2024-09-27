
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
import pickle
import matplotlib.pyplot as plt


def non_linearity(x, gamma):
    r = tf.math.square(x) / (gamma + tf.math.square(x) )
    return r

def inv_non_linearity(r, gamma):
    return tf.math.sqrt(r * gamma / (1 - r))


def letters_generator(d, noise=0.2, forced_letter=None):

    """
    :param d: number of letters generated
    :param noise: percentage of corrupted pixels
    :return: a dataset of noised letters and their labels
    """

    x_a = np.array([0, 0, 0, 0, 0, 0, 0,
                    0, 0, 1, 1, 1, 0, 0,
                    0, 1, 0, 0, 0, 1, 0,
                    0, 1, 1, 1, 1, 1, 0,
                    0, 1, 0, 0, 0, 1, 0,
                    0, 1, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 0, 0])

    x_b = np.array([0, 0, 0, 0, 0, 0, 0,
                    0, 1, 1, 1, 1, 0, 0,
                    0, 1, 0, 0, 0, 1, 0,
                    0, 1, 1, 1, 1, 0, 0,
                    0, 1, 0, 0, 0, 1, 0,
                    0, 1, 1, 1, 1, 0, 0,
                    0, 0, 0, 0, 0, 0, 0])

    x_c = np.array([0, 0, 0, 0, 0, 0, 0,
                    0, 0, 1, 1, 1, 1, 0,
                    0, 1, 0, 0, 0, 0, 0,
                    0, 1, 0, 0, 0, 0, 0,
                    0, 1, 0, 0, 0, 0, 0,
                    0, 0, 1, 1, 1, 1, 0,
                    0, 0, 0, 0, 0, 0, 0])

    alphabet = ['a', 'b', 'c']
    x_list = [x_a, x_b, x_c]

    letters = np.zeros((d, 49))
    letters_label = []

    if forced_letter is None:
        L = np.random.randint(3, size = d)
    else:
        L = np.ones(d) * alphabet.index(forced_letter)

    for i in range(d):
        l = int(L[i])
        x_sorted = x_list[l].copy()
        chosen_pixels = np.random.choice([0, 1], size=x_a.shape, p=[1 - noise, noise])
        x_sorted = x_sorted * (1 - chosen_pixels) + np.random.uniform(0, 1, x_sorted.shape) * chosen_pixels
        letters[i,:] = x_sorted.reshape(-1,)
        letters_label.append(l)

    return  letters , letters_label


class CVFR_model(tf.keras.Model):
    def __init__(self,
                size,
                attractors,
                deltat=0.1,
                max_steps=50,
                phi_trainable = None,
                eigenvalues = None,
                sigma=0.5,
                ):
        super(CVFR_model, self).__init__()

        """
        :param size: size of the network: must be equal to the input size if no transformation is operated before the model
        :param attractors: array of dimension size x number_of_classes. Every column is an attract for the dynamic
        :param deltat: integration step size
        :param max_steps: number of integration time step
        :param phi_trainable: if not None it defines the initial condition for the trainable part of the base
        :eigenvalue: if not None it define the initial condition for the trainable eigenvalues
        :sigma: std of the gaussian noise injected in the model 
        :return: a dataset of noised letters and their labels
        """

        self.size = tf.constant(size, dtype = 'float32')
        self.deltat = deltat
        self.max_steps = max_steps
        self.beta = tf.constant(1/np.sqrt(size), dtype = 'float32')
        self.gamma =1/(8*size)
        self.attractors = tf.constant(attractors, dtype='float32')
        self.xp = (self.beta + np.sqrt(self.beta**2 - 4* self.gamma))/2
        self.sigma = sigma
        self.n_classes = attractors.shape[1]

        if eigenvalues is None:
            self.eigenvalues = self.add_weight(
                name='trainable_eigenvalues',
                shape=(int(self.size - self.attractors.shape[1]),),
                initializer=tf.keras.initializers.RandomUniform(minval=-.4, maxval=.4),
                trainable=True,
                dtype=tf.float32
            )
        else:
            self.eigenvalues = tf.Variable(eigenvalues, dtype='float32')

        if phi_trainable is None:
            self.phi_trainable = self.add_weight(shape=(int(self.size), int(self.size - self.attractors.shape[1])),
                                              initializer=tf.keras.initializers.Orthogonal(gain=1.3),
                                             regularizer=tf.keras.regularizers.OrthogonalRegularizer(factor=0.004, mode='columns'),
                                              trainable=True,
                                              name='base_train')
        else:
            self.phi_trainable = tf.Variable(phi_trainable, dtype='float32')

    def compute_damping_factor(self,v, n_classes):
        r_attractors = inv_non_linearity(self.attractors, self.gamma)
        p =  tf.math.pow(tf.math.reduce_mean(tf.math.square(v -r_attractors[:,0]), axis=1), 1/n_classes)
        for i in range(1, self.attractors.shape[1]):
            p = p * tf.math.pow(tf.math.reduce_mean(tf.math.square(v - r_attractors[:,i]), axis=1), 1/n_classes)
        return tf.math.tanh(p)


    def call(self, inputs, max_steps = None, all_trajectory = False,  **kwargs):
        """
        :param inputs: initial condition
        :param max_steps: number of integration time step if it's None it is fixed to the value used during the initialization of the model
        :param all_trajectory: if it is True, the model returns all the trajectories, otherwise the model returns only the last state
        """


        self.phi = tf.concat([self.attractors, self.phi_trainable], axis=1)
        eigenvalues_total = tf.concat([tf.ones(self.attractors.shape[1]), self.eigenvalues], axis=0)
        diagonal = tf.linalg.diag(eigenvalues_total)

        w = tf.matmul(self.phi, tf.matmul(diagonal, tf.linalg.inv(self.phi)))

        if all_trajectory:
            prediction = tf.expand_dims(non_linearity(inputs, gamma=self.gamma), axis=2)


        damping_factor = self.compute_damping_factor(inputs, self.n_classes)
        noise = tf.transpose(tf.multiply(tf.transpose(tf.random.normal(tf.shape(inputs),0,self.sigma)), damping_factor ))

        x_dot = -inputs + self.beta * tf.matmul( non_linearity(inputs, gamma=self.gamma), tf.transpose(w))

        x = inputs + self.deltat * x_dot + tf.math.sqrt(self.deltat) * noise
        if all_trajectory:
            prediction = tf.concat([prediction, tf.expand_dims(non_linearity(x, gamma=self.gamma), axis=2)], axis=2)
        if max_steps is None:
            max_steps = self.max_steps

        for i in range(max_steps-1):

            damping_factor = self.compute_damping_factor(x, self.n_classes)
            noise = tf.transpose(tf.multiply(tf.transpose(tf.random.normal(tf.shape(inputs), 0, self.sigma)),damping_factor))

            x_dot = -x + self.beta * tf.matmul( non_linearity(x, gamma=self.gamma), tf.transpose(w))
            x = x + self.deltat * x_dot + tf.math.sqrt(self.deltat) * noise
            if all_trajectory:
                prediction = tf.concat([prediction, tf.expand_dims(non_linearity(x, gamma=self.gamma), axis=2)], axis=2)

        if all_trajectory:
            return prediction
        else:
            return non_linearity(x, gamma=self.gamma)


