import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import numpy as np
import tensorflow as tf


def reparameterize(mu,log_var):
    eps = tf.random.normal(log_var.shape)
    std = tf.exp(log_var)**0.5
    z = mu + std * eps
    return z

def Potential_y(y,t):
    potential_y = np.array(y)
    io = tf.where(t==1)
    io = np.array(io)
    io = io.reshape(-1,)
    for i in io:
        temp = potential_y[i][1]
        potential_y[i][1] = potential_y[i][0]
        potential_y[i][0] = temp
    return potential_y

def batch_generator(x, t, y, size):

    batch_idx = np.random.randint(0, x.shape[0], size)

    X_mb = x[batch_idx, :]
    T_mb = np.reshape(t[batch_idx], [size, 1])
    Y_mb = np.reshape(y[batch_idx], [size, 1])
    return X_mb, T_mb, Y_mb

def batch_generator_y(x , t, potential_y, size):
    batch_idx = np.random.randint(0,x.shape[0],size)
    potential_y = np.array(potential_y)
    X_mb = x[batch_idx,:]
    T_mb = np.reshape(t[batch_idx],[size, 1])
    Y_mb = np.reshape(potential_y[batch_idx,:],[size,2])
    return X_mb, T_mb, Y_mb