import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import tensorflow.keras.layers as layers
import numpy as np
from utils import *
from metrics import *


def vganite_syn_nozc(train_x,train_t,train_y,train_potential_y,test_x,test_potential_y,y_test,test_t,parameters):

    batch_size = parameters['batch_size']
    iterations = parameters['iteration']
    h_dim = parameters['h_dim']
    learning_rate = parameters['learning_rate']
    x_dim = parameters['x_dim']
    class VAE(tf.keras.Model):
        def __init__(self):
            super(VAE, self).__init__()
            #Encoder网络
            self.fc1 = layers.Dense(100,activation=tf.nn.relu)
            self.dropout1 = tf.keras.layers.Dropout(0.5)
            self.fc1_1 = layers.Dense(100)
            self.fc12 = layers.Dense(100,activation=tf.nn.relu)
            self.dropout2 = tf.keras.layers.Dropout(0.5)
            self.fc1_2 = layers.Dense(100)



            self.fc3 = layers.Dense(100,activation=tf.nn.relu)
            self.dropout5 = tf.keras.layers.Dropout(0.5)
            self.fc3_1 = layers.Dense(100)
            self.fc32 = layers.Dense(100,activation=tf.nn.relu)
            self.dropout6 = tf.keras.layers.Dropout(0.5)
            self.fc3_2 = layers.Dense(100)


            #Decoder网络
            self.fc4 = layers.Dense(200,activation=tf.nn.relu)
            self.fc4_1 = layers.Dense(100,activation=tf.nn.relu)
            self.fc41 = layers.Dense(x_dim)



            self.fc5 = layers.Dense(50,activation=tf.nn.relu)
            self.fc5_1 = layers.Dense(10,activation=tf.nn.relu)
            self.fc51 = layers.Dense(1,activation=tf.nn.sigmoid)

            self.fc8 = layers.Dense(50,activation=tf.nn.relu)
            self.fc8_1 = layers.Dense(10,activation=tf.nn.relu)
            self.fc81 = layers.Dense(1,activation=tf.nn.sigmoid)



            # generator网络
            #generator网络
            self.G_h1 = tf.keras.layers.Dense(units=h_dim, activation=tf.nn.relu)
            self.dropout7 = layers.Dropout(0)
            self.G_h2 = tf.keras.layers.Dense(units=h_dim, activation=tf.nn.relu)
            self.G_h31 = tf.keras.layers.Dense(units=h_dim, activation=tf.nn.relu)
            self.G_h41 = tf.keras.layers.Dense(units=h_dim, activation=tf.nn.relu)
            self.G_logit1 = tf.keras.layers.Dense(units=1)
            self.G_logit2 = tf.keras.layers.Dense(units=1)

        def encoder(self,x):
            ht = self.fc1(x)
            ht = self.dropout1(ht)
            mu_t = tf.nn.relu(self.fc1_1(ht))
            log_var_t = self.fc12(x)
            log_var_t = self.dropout2(log_var_t)
            log_var_t = self.fc1_2(log_var_t)
            hy = self.fc3(x)
            hy = self.dropout5(hy)
            mu_y = tf.nn.relu(self.fc3_1(hy))
            log_var_y = self.fc32(x)
            log_var_y = self.dropout6(log_var_y)
            log_var_y = self.fc3_2(log_var_y)

            return mu_t,log_var_t,mu_y,log_var_y

        def decoder(self,z_t,z_y):
            Z = tf.concat([z_t,z_y],axis=1)
            Z = self.fc4(Z)
            x = self.fc4_1(Z)
            x = self.fc41(x)
            return x

        def decoder_t(self,z_t):
            t_hat = self.fc5(z_t)
            t_hat = self.fc5_1(t_hat)
            t_hat = self.fc51(t_hat)
            return t_hat

        def treatment(self,z_y):
            t_hat = self.fc8(z_y)
            t_hat = self.fc8_1(t_hat)
            t_hat = self.fc81(t_hat)
            return t_hat

        def generator(self,z_y):
            inputs = z_y
            G_h1 = self.G_h1(inputs)
            G_h1 = self.dropout7(G_h1)
            G_h2 = self.G_h2(G_h1)
            G_h31 = self.G_h31(G_h2)
            G_h41 = self.G_h41(G_h2)
            G_logit1 = self.G_logit1(G_h31)
            G_logit2 = self.G_logit2(G_h41)
            G_logit = tf.concat([G_logit1,G_logit2],axis=1)#
            return G_logit#[Y(0),Y(1)]


        def call(self,inputs,t,y,training=None):
            mu_t,log_var_t,mu_y,log_var_y = self.encoder(inputs)
            z_t = reparameterize(mu_t,log_var_t)
            z_y = reparameterize(mu_y,log_var_y)
            z_t = z_t/tf.sqrt(tf.reduce_sum(tf.square(z_t),axis=1,keepdims=True))
            z_y = z_y/tf.sqrt(tf.reduce_sum(tf.square(z_y),axis=1,keepdims=True))
            x_hat = self.decoder(z_t,z_y)
            t_hat = self.decoder_t(z_t)
            y_hat = self.generator(z_y)
            treat_t_hat = self.treatment(z_y)

            return z_y,x_hat,t_hat,y_hat,mu_t,log_var_t,mu_y,log_var_y,treat_t_hat


    class discriminator(tf.keras.Model):

        def __init__(self):
            super().__init__()
            self.D_h1 = tf.keras.layers.Dense(units=10,activation=tf.nn.relu)
            self.D_h2 = tf.keras.layers.Dense(units=5,activation=tf.nn.relu)
            self.D_logit = tf.keras.layers.Dense(units=1)
        def call(self,z_y,t,y,y_hat):
            #Concatenate factual and counterfactual outcomes
            input0 = (1. - t) * y + t * tf.reshape(y_hat[:, 0], [-1, 1])  # if t = 0
            input1 = t * y + (1. - t) * tf.reshape(y_hat[:, 1], [-1, 1])  # if t = 1
            inputs = tf.concat([z_y,input0,input1],axis=1)#[X,Y(0),Y(1)]
            D_h1 = self.D_h1(inputs)
            D_h2 = self.D_h2(D_h1)
            D_logit = self.D_logit(D_h2)
            return D_logit
    class inference(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.I_h1 = tf.keras.layers.Dense(units=h_dim,activation=tf.nn.relu)
            self.I_h2 = tf.keras.layers.Dense(units=h_dim,activation=tf.nn.relu)
            self.I_h32 = tf.keras.layers.Dense(units=h_dim, activation=tf.nn.relu)
            self.I_logit0 = tf.keras.layers.Dense(units=2)
            # self.I_logit1 = tf.keras.layers.Dense(units=1)#[Y(0),Y(1)]
        def call(self,x):
            I_h1 = self.I_h1(x)
            I_h2 = self.I_h2(I_h1)
            I_h31 = self.I_h32(I_h2)
            I_logit = self.I_logit0(I_h31)

            return I_logit#[Y(0),Y(1)]


    model = VAE()
    Discriminator = discriminator()
    Inference = inference()


    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    optimizer_d = tf.keras.optimizers.Adam(lr=learning_rate)
    optimizer_i = tf.keras.optimizers.Adam(lr=learning_rate)
    print("Strat training")
    for epoch in range(1):
        for it in range(iterations):
            for _ in range(2):
                with tf.GradientTape() as tape:
                    x, t, y = batch_generator(train_x, train_t, train_y, batch_size)
                    x = tf.cast(x, tf.float32)
                    t = tf.cast(t, tf.float32)
                    y = tf.cast(y, tf.float32)
                    z_y,x_hat,t_hat,y_hat,mu_t,log_var_t,mu_y,log_var_y,treat_t_hat= model(x,t,y)
                    Y_tilde = tf.nn.sigmoid(y_hat)
                    D_logit = Discriminator(z_y,t,y,Y_tilde)
                    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=D_logit))
                D_grads = tape.gradient(D_loss, Discriminator.trainable_variables)
                optimizer_d.apply_gradients(grads_and_vars=zip(D_grads, Discriminator.trainable_variables))

            with tf.GradientTape() as tape:
                x, t, y = batch_generator(train_x, train_t, train_y, batch_size)
                x = tf.cast(x, tf.float32)
                t = tf.cast(t, tf.float32)
                y = tf.cast(y, tf.float32)
                p_t = tf.cast(tf.reduce_sum(t) / t.shape[0], tf.float32)
                w_t = t / (2. * p_t)
                w_c = (1. - t) / (2. * (1. - p_t))
                z_y, x_hat, t_hat, y_hat, mu_t, log_var_t, mu_y, log_var_y ,treat_t_hat= model(x,t,y)
                treat_t_hat = tf.nn.sigmoid(treat_t_hat)
                pi_0 = tf.cast(treat_t_hat, tf.float32)
                sample_weight = 1. * (1. + (1. - pi_0) / pi_0 * (p_t / (1. - p_t)) ** (2. * t - 1.)) * (w_t + w_c)
                treat_t_hat_loss = - tf.reduce_mean(tf.multiply(t, tf.math.log(treat_t_hat)) + tf.multiply(1.0-t, tf.math.log(1.0-treat_t_hat)) )
                rec_x_loss = tf.reduce_mean(tf.square(x_hat-x))
                rec_t_loss = - tf.reduce_mean(tf.multiply(t, tf.math.log(t_hat)) + tf.multiply(1.0-t, tf.math.log(1.0-t_hat)) )

                G_loss_Factual = tf.reduce_mean(sample_weight * tf.square((t * tf.reshape(y_hat[:, 1], [-1, 1])) + \
                                                          ((1. - t) * tf.reshape(y_hat[:, 0], [-1, 1])) - y))

                G_loss_GAN = - D_loss
                G_loss = G_loss_GAN + 2 * G_loss_Factual
                kl_div_t = -0.5 * (log_var_t + 1 - mu_t ** 2 - tf.exp(log_var_t))
                kl_div_y = -0.5 * (log_var_y + 1 - mu_y ** 2 - tf.exp(log_var_y))
                kl_div_t = tf.reduce_sum(kl_div_t) / x.shape[0]
                kl_div_y = tf.reduce_sum(kl_div_y) / x.shape[0]

                loss = 1e-4*rec_x_loss + 1e-4*rec_t_loss +1e-4*treat_t_hat_loss+ 1e-4*(kl_div_t+kl_div_y) + G_loss
            grads = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))


            if it % 100 == 0:
                x, t, y = train_x, train_t, train_y
                x = tf.cast(x, tf.float32)
                t = tf.cast(t, tf.float32)
                y = tf.cast(y, tf.float32)
                z_y, x_hat, t_hat,y_hat, mu_t, log_var_t,  mu_y, log_var_y,treat_t_hat = model(x, t, y)
                y_hat = tf.cast(y_hat,tf.float32)
                train_potential_y = tf.cast(train_potential_y,tf.float32)
                pehe = PEHE(train_potential_y, y_hat)
                ate = ATE(train_potential_y, y_hat)
                train_ate_t = np.mean(train_potential_y[:, 1]) - np.mean(train_potential_y[:, 0])
                train_ate_hat = np.mean((y_hat[:, 1]) - (y_hat[:, 0]))

                print("epoch:"+str(epoch)+" iteration:" + str(it) + "/" + str(iterations) + " loss:" + str(
                    np.round(loss, 4)) + " f_loss:"+ str(np.round(G_loss_Factual, 4)) +" D_loss:"+str(np.round(D_loss,4))+ " Train_PEHE:" + str(np.round(pehe, 4)) + " Train_ATE:" + str(
                    np.round(ate, 4))
                      + " ATE_T:" + str(np.round(train_ate_t, 4)) + " ATE_hat:" + str(np.round(train_ate_hat, 4)))

                # x, t, y = test_x, test_t, y_test
                # x = tf.cast(x, tf.float32)
                # t = tf.cast(t, tf.float32)
                # y = tf.cast(y, tf.float32)
                # z_y, x_hat, t_hat, y_hat, mu_t, log_var_t,  mu_y, log_var_y = model(x,t,y)
                # y_hat = tf.cast(y_hat, tf.float32)
                # test_potential_y = tf.cast(test_potential_y,tf.float32)
                # test_pehe = PEHE(test_potential_y, y_hat)
                # test_ate = ATE(test_potential_y, y_hat)
                # test_ate_t = np.mean(test_potential_y[:, 1]) - np.mean(test_potential_y[:, 0])
                # test_ate_hat = np.mean((y_hat[:, 1]) - (y_hat[:, 0]))
                #
                # print("Test_PEHE:" + str(np.round(test_pehe, 4)) + " Test_ATE:" + str(
                #     np.round(test_ate, 4))
                #       + " ATE_T:" + str(np.round(test_ate_t, 4)) + " ATE_hat:" + str(np.round(test_ate_hat, 4)))


    print("Start training Inference")
    x, t, y = train_x, train_t, train_y
    x = tf.cast(x, tf.float32)
    t = tf.cast(t, tf.float32)
    y = tf.cast(y, tf.float32)
    z_y, x_hat, t_hat, y_hat_factual, mu_t, log_var_t,  mu_y, log_var_y ,treat_t_hat= model(x,t,y)
    for it in range(iterations):
        with tf.GradientTape() as tape:
            x, t, y_hat = batch_generator_y(train_x, train_t, y_hat_factual, batch_size)
            Y_hat_logit = Inference(x)
            I_loss1 = tf.reduce_mean(tf.square(t*Y_hat_logit[:,1] - (t*y_hat[:,1])))
            I_loss2 = tf.reduce_mean(tf.square((1-t)*Y_hat_logit[:,0]-((1-t)*y_hat[:,0])))
            I_loss = I_loss1+I_loss2
            I_grads = tape.gradient(I_loss, Inference.trainable_variables)
            optimizer_i.apply_gradients(grads_and_vars=zip(I_grads, Inference.trainable_variables))

        if it % 100 == 0:
            print('Iteration: ' + str(it) + '/' + str(iterations) +
                  ', I loss: ' + str(np.round(I_loss, 4)))


    test_y_hat = Inference(test_x)
    return test_y_hat










