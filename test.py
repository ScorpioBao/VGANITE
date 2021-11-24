import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import tensorflow.keras.layers as layers
from utils import *
from metrics import *



def vganite_twin(train_x,train_t,train_y,train_potential_y,test_x,test_potential_y,y_test,test_t,parameters):

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

            self.fc2 = layers.Dense(100,activation=tf.nn.relu)
            self.dropout3 = tf.keras.layers.Dropout(0.5)
            self.fc2_1 = layers.Dense(100)
            self.fc22 = layers.Dense(100,activation=tf.nn.relu)
            self.dropout4 = tf.keras.layers.Dropout(0.5)
            self.fc2_2 = layers.Dense(100)

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


            hc = self.fc2(x)
            hc = self.dropout3(hc)
            mu_c = tf.nn.relu(self.fc2_1(hc))
            log_var_c = self.fc22(x)
            log_var_c = self.dropout4(log_var_c)
            log_var_c = self.fc2_2(log_var_c)

            hy = self.fc3(x)
            hy = self.dropout5(hy)
            mu_y = tf.nn.relu(self.fc3_1(hy))
            log_var_y = self.fc32(x)
            log_var_y = self.dropout6(log_var_y)
            log_var_y = self.fc3_2(log_var_y)

            return mu_t,log_var_t,mu_c,log_var_c,mu_y,log_var_y

        def decoder(self,z_t,z_c,z_y):
            Z = tf.concat([z_t,z_c,z_y],axis=1)
            Z = self.fc4(Z)
            x = self.fc4_1(Z)
            x = self.fc41(x)
            return x

        def decoder_t(self,z_t,z_c):
            t_hat = self.fc5(tf.concat([z_t,z_c],axis=1))
            t_hat = self.fc5_1(t_hat)
            t_hat = self.fc51(t_hat)
            return t_hat
        #
        # def treatment(self,z_c):
        #     t_hat = self.fc8(z_c)
        #     t_hat = self.fc8_1(t_hat)
        #     t_hat = self.fc81(t_hat)
        #     return t_hat


        # def decoder_y(self,z_c,z_y,t):
        #     rep = tf.concat([z_c,z_y],axis=1)
        #     i0 = tf.cast(tf.where(t < 1)[:, 0], tf.int32)
        #     i1 = tf.cast(tf.where(t > 0)[:, 0], tf.int32)
        #     rep0 = tf.gather(rep, i0)
        #     rep1 = tf.gather(rep, i1)
        #     y0_hat = self.fc6(rep0)
        #     y0_hat = self.dropout7(y0_hat)
        #     y0_hat = self.fc61(y0_hat)
        #     y0_hat = self.fc62(y0_hat)
        #     y1_hat = self.fc7(rep1)
        #     y1_hat = self.dropout8(y1_hat)
        #     y1_hat = self.fc71(y1_hat)
        #     y1_hat = self.fc72(y1_hat)
        #     y = tf.dynamic_stitch([i0,i1],[y0_hat,y1_hat])
        #     return y
        # def generator(self,z_c,z_y):
        #     inputs = tf.concat([z_c,z_y],axis=1)
        #     G_h1 = self.G_h1(inputs)
        #     G_h1 = self.dropout7(G_h1)
        #     G_h2 = self.G_h2(G_h1)
        #     G_h31 = self.G_h31(G_h2)
        #     G_h41 = self.G_h41(G_h2)
        #     G_logit1 = self.G_logit1(G_h31)
        #     G_logit2 = self.G_logit2(G_h41)
        #     G_logit = tf.concat([G_logit1,G_logit2],axis=1)#
        #     G_logit = tf.nn.sigmoid(G_logit)
        #     return G_logit#[Y(0),Y(1)]


        def call(self,inputs,t,y,training=None):
            mu_t,log_var_t,mu_c,log_var_c,mu_y,log_var_y = self.encoder(inputs)
            z_t = reparameterize(mu_t,log_var_t)
            z_c = reparameterize(mu_c,log_var_c)
            z_y = reparameterize(mu_y,log_var_y)
            z_t = z_t/tf.sqrt(tf.reduce_sum(tf.square(z_t),axis=1,keepdims=True))
            z_c = z_c/tf.sqrt(tf.reduce_sum(tf.square(z_c),axis=1,keepdims=True))
            z_y = z_y/tf.sqrt(tf.reduce_sum(tf.square(z_y),axis=1,keepdims=True))
            x_hat = self.decoder(z_t,z_c,z_y)
            t_hat = self.decoder_t(z_t,z_c)
            # y_hat = self.generator(z_c,z_y)
            # treat_t_hat = self.treatment(z_c)

            return z_c,z_y,x_hat,t_hat,mu_t,log_var_t,mu_c,log_var_c,mu_y,log_var_y


    # class discriminator(tf.keras.Model):
    #
    #     def __init__(self):
    #         super().__init__()
    #         self.D_h1 = tf.keras.layers.Dense(units=10,activation=tf.nn.relu)
    #         self.D_h2 = tf.keras.layers.Dense(units=5,activation=tf.nn.relu)
    #         self.D_logit = tf.keras.layers.Dense(units=1)
    #     def call(self,z_c,z_y,t,y,y_hat):
    #         #Concatenate factual and counterfactual outcomes
    #         input0 = (1. - t) * y + t * tf.reshape(y_hat[:, 0], [-1, 1])  # if t = 0
    #         input1 = t * y + (1. - t) * tf.reshape(y_hat[:, 1], [-1, 1])  # if t = 1
    #         inputs = tf.concat([z_c,z_y,input0,input1],axis=1)#[X,Y(0),Y(1)]
    #         D_h1 = self.D_h1(inputs)
    #         D_h2 = self.D_h2(D_h1)
    #         D_logit = self.D_logit(D_h2)
    #         return D_logit
    # class inference(tf.keras.Model):
    #     def __init__(self):
    #         super().__init__()
    #         self.I_h1 = tf.keras.layers.Dense(units=h_dim,activation=tf.nn.relu)
    #         self.I_h2 = tf.keras.layers.Dense(units=h_dim,activation=tf.nn.relu)
    #         self.I_h32 = tf.keras.layers.Dense(units=h_dim, activation=tf.nn.relu)
    #         self.I_logit0 = tf.keras.layers.Dense(units=2)
    #         # self.I_logit1 = tf.keras.layers.Dense(units=1)#[Y(0),Y(1)]
    #     def call(self,x):
    #         I_h1 = self.I_h1(x)
    #         I_h2 = self.I_h2(I_h1)
    #         I_h31 = self.I_h32(I_h2)
    #         I_logit = self.I_logit0(I_h31)
    #         # I_logit1 = self.I_logit1(I_h32)
    #         # I_logit = tf.concat([I_logit0,I_logit1],axis=1)
    #         return I_logit#[Y(0),Y(1)]


    model = VAE()
    # Discriminator = discriminator()
    # Inference = inference()
    # initial_learning_rate = 0.01
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate,
    #     decay_steps=1000,
    #     decay_rate=0.96,
    #     staircase=True)

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    # optimizer_d = tf.keras.optimizers.Adam(lr=learning_rate)
    # optimizer_i = tf.keras.optimizers.Adam(lr=learning_rate)
    print("Strat training")
    for epoch in range(2):
        for it in range(iterations):
            # for _ in range(2):
            #     with tf.GradientTape() as tape:
            #         x, t, y = batch_generator(train_x, train_t, train_y, batch_size)
            #         x = tf.cast(x, tf.float32)
            #         t = tf.cast(t, tf.float32)
            #         y = tf.cast(y, tf.float32)
            #         z_c,z_y,x_hat,t_hat,y_hat,mu_t,log_var_t,mu_c,log_var_c,mu_y,log_var_y,treat_t_hat= model(x,t,y)
            #         Y_tilde = tf.nn.sigmoid(y_hat)
            #         D_logit = Discriminator(z_c,z_y,t,y,Y_tilde)
            #         D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=D_logit))
            #     D_grads = tape.gradient(D_loss, Discriminator.trainable_variables)
            #     optimizer_d.apply_gradients(grads_and_vars=zip(D_grads, Discriminator.trainable_variables))

            with tf.GradientTape() as tape:
                x, t, y = batch_generator(train_x, train_t, train_y, batch_size)
                x = tf.cast(x, tf.float32)
                t = tf.cast(t, tf.float32)
                y = tf.cast(y, tf.float32)
                # p_t = tf.cast(tf.reduce_sum(t) / t.shape[0], tf.float32)
                # w_t = t / (2. * p_t)
                # w_c = (1. - t) / (2. * (1. - p_t))
                z_c, z_y, x_hat, t_hat, mu_t, log_var_t, mu_c, log_var_c, mu_y, log_var_y,  = model(x,t,y)

                rec_x_loss = tf.reduce_mean(tf.abs(x - x_hat))
                t_hat = tf.nn.sigmoid(t_hat)
                rec_t_loss = - tf.reduce_mean(t * tf.math.log(t_hat)+(1-t)* tf.math.log(1-t_hat))

                kl_div_t = -0.5 * (log_var_t + 1 - mu_t ** 2 - tf.exp(log_var_t))
                kl_div_c = -0.5 * (log_var_c + 1 - mu_c ** 2 - tf.exp(log_var_c))
                kl_div_y = -0.5 * (log_var_y + 1 - mu_y ** 2 - tf.exp(log_var_y))
                kl_div_t = tf.reduce_sum(kl_div_t) / x.shape[0]
                kl_div_c = tf.reduce_sum(kl_div_c) / x.shape[0]
                kl_div_y = tf.reduce_sum(kl_div_y) / x.shape[0]

                loss = rec_x_loss +(kl_div_t+kl_div_c+kl_div_y)+rec_t_loss
            grads = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))


            if it % 100 == 0:


                print("epoch:"+str(epoch)+" iteration:" + str(it) + "/" + str(iterations) + " loss:" + str(
                    np.round(loss, 4)) + " f_loss:"+ str(np.round(rec_x_loss, 4)) +" KL:"+str(np.round((kl_div_t+kl_div_c+kl_div_y),4)) +" t_loss:"+str(np.round(rec_t_loss,4)))

    return 0










