import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
import numpy as np
import argparse

from vganite_ihdp import vganite_ihdp
from vganite_twin import vganite_twin
from vganite_syn import vganite_syn
from vganite_syn_nozt import vganite_syn_nozt
from vganite_syn_nozc import vganite_syn_nozc
from vganite_syn_nozy import vganite_syn_nozy


from  metrics import PEHE,ATE
from datasets import IHDP,data_loading_twin,Syn



def main(args):
    parameters = dict()
    parameters['data_name'] = args.data_name
    parameters['batch_size'] = args.batch_size
    parameters['iteration'] = args.iteration
    parameters['learning_rate'] = args.learning_rate
    parameters['h_dim'] = args.h_dim
    parameters['x_dim'] = args.x_dim


    if args.data_name =="ihdp":
        train, test, contfeats, binfeats = IHDP(path="IHDP", reps=12)
        (x_train, t_train, y_train), true_ite_train,train_potential_y = train
        (x_test, t_test, y_test), true_ite_test,test_potential_y = test

        X_train = x_train
        t_train = np.reshape(t_train,(673,1))
        y_train = np.reshape(y_train,(673,1))

        X_test = x_test
        t_test = np.reshape(t_test,(74,1))
        y_test = np.reshape(y_test,(74,1))
        y_hat = vganite_ihdp(X_train, t_train, y_train, train_potential_y, X_test, test_potential_y,y_test,t_test, parameters)
        y_hat = tf.cast(y_hat,tf.float32)
        test_potential_y = tf.cast(test_potential_y,tf.float32)
        ate_t = np.mean((test_potential_y[:, 1]) - (test_potential_y[:, 0]))
        ate_hat = np.mean((y_hat[:, 1]) - (y_hat[:, 0]))
        print('Test_T_ATE',ate_t)
        print('Test_hat_ATE', ate_hat)
        print("Test_PEHE:", PEHE(test_potential_y, y_hat))
        print("Test_ATE:", ATE(test_potential_y, y_hat))
    if args.data_name =="twin":
        train_x, train_t, train_y, train_potential_y, test_x, test_t, test_y, test_potential_y = data_loading_twin(0.8)
        X_train = train_x
        t_train = np.reshape(train_t,(9120,1))
        y_train = np.reshape(train_y,(9120,1))

        X_test = test_x
        t_test = np.reshape(test_t,(2280,1))
        y_test = np.reshape(test_y,(2280,1))
        y_hat = vganite_twin(X_train, t_train, y_train, train_potential_y, X_test, test_potential_y, y_test, t_test,
                        parameters)
        y_hat = tf.cast(y_hat, tf.float32)
        test_potential_y = tf.cast(test_potential_y, tf.float32)
        ate_t = np.mean((test_potential_y[:, 1]) - (test_potential_y[:, 0]))
        ate_hat = np.mean((y_hat[:, 1]) - (y_hat[:, 0]))
        print('Test_T_ATE', ate_t)
        print('Test_hat_ATE', ate_hat)
        print("Test_PEHE:", PEHE(test_potential_y, y_hat))
        print("Test_ATE:", ATE(test_potential_y, y_hat))
    if args.data_name =="syn":
        train, test = Syn(path = './data/Syn_1.0_1.0_0/8_8_4',reps=1)
        (x_train, t_train, y_train), true_ite_train, train_potential_y = train
        (x_test, t_test, y_test), true_ite_test, test_potential_y = test

        X_train = x_train
        t_train = np.reshape(t_train, (14999, 1))
        y_train = np.reshape(y_train, (14999, 1))

        X_test = x_test
        t_test = np.reshape(t_test, (4999, 1))
        y_test = np.reshape(y_test, (4999, 1))
        y_hat = vganite_syn(X_train, t_train, y_train, train_potential_y, X_test, test_potential_y, y_test, t_test,
                             parameters)
        y_hat = tf.cast(y_hat, tf.float32)
        test_potential_y = tf.cast(test_potential_y, tf.float32)
        ate_t = np.mean((test_potential_y[:, 1]) - (test_potential_y[:, 0]))
        ate_hat = np.mean((y_hat[:, 1]) - (y_hat[:, 0]))
        print('Test_T_ATE', ate_t)
        print('Test_hat_ATE', ate_hat)
        print("Test_PEHE:", PEHE(test_potential_y, y_hat))
        print("Test_ATE:", ATE(test_potential_y, y_hat))
    if args.data_name =="syn_no":
        train, test = Syn(path = './data/Syn_1.0_1.0_0/8_8_4',reps=1)
        (x_train, t_train, y_train), true_ite_train, train_potential_y = train
        (x_test, t_test, y_test), true_ite_test, test_potential_y = test

        X_train = x_train
        t_train = np.reshape(t_train, (14999, 1))
        y_train = np.reshape(y_train, (14999, 1))

        X_test = x_test
        t_test = np.reshape(t_test, (4999, 1))
        y_test = np.reshape(y_test, (4999, 1))
        y_hat = vganite_syn_nozc(X_train, t_train, y_train, train_potential_y, X_test, test_potential_y, y_test, t_test,
                             parameters)
        y_hat = tf.cast(y_hat, tf.float32)
        test_potential_y = tf.cast(test_potential_y, tf.float32)
        ate_t = np.mean((test_potential_y[:, 1]) - (test_potential_y[:, 0]))
        ate_hat = np.mean((y_hat[:, 1]) - (y_hat[:, 0]))
        print('Test_T_ATE', ate_t)
        print('Test_hat_ATE', ate_hat)
        print("Test_PEHE:", PEHE(test_potential_y, y_hat))
        print("Test_ATE:", ATE(test_potential_y, y_hat))

if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['twin', 'ihdp','syn','syn_no'],
        default='ihdp',
        type=str)
    parser.add_argument(
        '--x_dim',
        choices=[13,17,21,25,30],
        default=25,
        type=int)
    parser.add_argument(
        '--iteration',
        help='Training iterations (should be optimized)',
        default=300,
        type=int)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch (should be optimized)',
        default=1024,
        type=int)
    parser.add_argument(
        '--learning_rate',
        choices=[0.001,0.0001],
        default=0.001,
        type=float
    )
    parser.add_argument(  # s in paper
        '--h_dim',
        help='hidden state dimensions (should be optimized)',
        default=5,
        type=int)
    args = parser.parse_args()
    main(args)
