import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import tensorflow as tf
import numpy as np
from scipy.special import expit
from utils import batch_generator
def IHDP(path = './IHDP',reps=1):
    path_data = path
    replications = reps
    # which features are binary
    binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    # which features are continuous
    contfeats = [i for i in range(25) if i not in binfeats]

    data = np.loadtxt(path_data + '/ihdp_npci_train_' + str(replications) + '.csv', delimiter=',', skiprows=1)
    t, y = data[:, 0], data[:, 1][:, np.newaxis]
    mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
    true_ite = mu_1 - mu_0
    train_potential_y = tf.concat([mu_0, mu_1], axis=1)
    x[:, 13] -= 1
    # perm = binfeats + contfeats
    # x = x[:, perm]

    x = np.array(x)
    y = np.array(y).squeeze()
    t = np.array(t).squeeze()
    train = (x, t, y), true_ite, train_potential_y

    data_test = np.loadtxt(path_data + '/ihdp_npci_test_' + str(replications) + '.csv', delimiter=',', skiprows=1)
    t_test, y_test = data_test[:, 0][:, np.newaxis], data_test[:, 1][:, np.newaxis]
    mu_0_test, mu_1_test, x_test = data_test[:, 3][:, np.newaxis], data_test[:, 4][:, np.newaxis], data_test[:, 5:]
    x_test[:, 13] -= 1
    # x_test = x_test[:, perm]

    x_test = np.array(x_test)
    y_test = np.array(y_test).squeeze()
    t_test = np.array(t_test).squeeze()

    true_ite_test = mu_1_test - mu_0_test
    test_potential_y = tf.concat([mu_0_test, mu_1_test], axis=1)
    test = (x_test, t_test, y_test), true_ite_test, test_potential_y
    return train, test, contfeats, binfeats



def data_loading_twin(train_rate=0.8):
  """Load twins data.

  Args:
    - train_rate: the ratio of training data

  Returns:
    - train_x: features in training data
    - train_t: treatments in training data
    - train_y: observed outcomes in training data
    - train_potential_y: potential outcomes in training data
    - test_x: features in testing data
    - test_potential_y: potential outcomes in testing data
  """

  # Load original data (11400 patients, 30 features, 2 dimensional potential outcomes)
  ori_data = np.loadtxt("Twin/Twin_data.csv", delimiter=",", skiprows=1, encoding='utf-8')

  # Define features
  x = ori_data[:, :30]
  # x_mean = np.reshape(np.mean(x, axis=1), (11400, 1))
  # x_std = np.reshape(np.std(x, axis=1), (11400, 1))
  # x = (x - x_mean) / x_std
  no, dim = x.shape

  # Define potential outcomes
  potential_y = ori_data[:, 30:]
  # Die within 1 year = 1, otherwise = 0
  potential_y = np.array(potential_y < 9999, dtype=float)

  ## Assign treatment
  coef = np.random.uniform(-0.001, 0.001, size=[dim, 1])
  prob_temp = expit(np.matmul(x, coef) + np.random.normal(0, 0.001, size=[no, 1]))

  prob_t = prob_temp / (2 * np.mean(prob_temp))
  prob_t[prob_t > 1] = 1

  t = np.random.binomial(1, prob_t, [no, 1])
  t = t.reshape([no, ])

  ## Define observable outcomes
  y = np.zeros([no, 1])
  y = np.transpose(t) * potential_y[:, 1] + np.transpose(1 - t) * potential_y[:, 0]
  y = np.reshape(np.transpose(y), [no, ])

  ## Train/test division
  idx = np.random.permutation(no)
  train_idx = idx[:int(train_rate * no)]
  test_idx = idx[int(train_rate * no):]

  train_x = x[train_idx, :]
  train_t = t[train_idx]
  train_y = y[train_idx]
  train_potential_y = potential_y[train_idx, :]

  test_x = x[test_idx, :]
  test_t = t[test_idx]
  test_y = y[test_idx]
  test_potential_y = potential_y[test_idx, :]


  test_potential_y = tf.convert_to_tensor(test_potential_y, dtype=tf.float32)

  return train_x, train_t, train_y, train_potential_y, test_x, test_t, test_y, test_potential_y


def Syn(path = './data/Syn_1.0_1.0_0/8_8_8',reps=1):
    path_data = path
    replications = reps

    data = np.loadtxt(path_data + '/4_' + str(replications) + '.csv', delimiter=',', skiprows=1)
    t, y = data[:, 0], data[:, 1][:, np.newaxis]
    mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
    true_ite = mu_1 - mu_0
    train_potential_y = tf.concat([mu_0, mu_1], axis=1)
    # perm = binfeats + contfeats
    # x = x[:, perm]

    x = np.array(x)
    y = np.array(y).squeeze()
    t = np.array(t).squeeze()
    train = (x, t, y), true_ite, train_potential_y

    data_test = np.loadtxt(path_data + '/4_test.csv', delimiter=',', skiprows=1)
    t_test, y_test = data_test[:, 0][:, np.newaxis], data_test[:, 1][:, np.newaxis]
    mu_0_test, mu_1_test, x_test = data_test[:, 3][:, np.newaxis], data_test[:, 4][:, np.newaxis], data_test[:, 5:]

    x_test = np.array(x_test)
    y_test = np.array(y_test).squeeze()
    t_test = np.array(t_test).squeeze()

    true_ite_test = mu_1_test - mu_0_test
    test_potential_y = tf.concat([mu_0_test, mu_1_test], axis=1)
    test = (x_test, t_test, y_test), true_ite_test, test_potential_y
    return train, test

# train, test = Syn(path = './data/Syn_1.0_1.0_0/8_8_8',reps=1)
# (x_train, t_train, y_train), true_ite_train,train_potential_y = train
# (x_test, t_test, y_test), true_ite_test, test_potential_y = test
# print(train_potential_y.shape)
# print(x_test.shape)