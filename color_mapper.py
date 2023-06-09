import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tqdm import tqdm
from utils import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
RESULTS_COLOR_DIR = os.path.join(RESULTS_DIR, 'color-mapper')

class ColorMapper:
    def __init__(self, model_name, fourier_max_freq=0, hidden_size=64, layer_num=8, batch_size=2048,
        activation='elu', learning_rate=0.0005, use_siren=True, use_sdf=False):
        self.model_name = model_name
        self.fourier_max_freq = fourier_max_freq
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.batch_size = batch_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.use_siren = use_siren
        self.use_sdf = use_sdf

    def create_model(self):
        point_in = Input(shape=(3,))

        if self.use_siren:
            if (self.fourier_max_freq > 0):
                x = Lambda(get_fourier_features, arguments={'fourier_max_freq': self.fourier_max_freq})(point_in)
                x = Dense(
                    units = self.hidden_size,
                    activation = tf.math.sin,
                    kernel_initializer = 'he_uniform',
                    bias_initializer = 'he_uniform')(x)
            else:
                x = Dense(
                    units = self.hidden_size,
                    activation = tf.math.sin,
                    kernel_initializer = 'he_uniform',
                    bias_initializer = 'he_uniform')(point_in)

            for _ in range(self.layer_num - 1):
                x = Dense(
                    units = self.hidden_size,
                    activation = tf.math.sin,
                    kernel_initializer = 'he_uniform',
                    bias_initializer = 'he_uniform')(x)

        else:
            if (self.fourier_max_freq > 0):
                x = Lambda(get_fourier_features, arguments={'fourier_max_freq': self.fourier_max_freq})(point_in)
                x = Dense(
                    units = self.hidden_size,
                    activation = self.activation)(x)
            else:
                x = Dense(
                    units = self.hidden_size,
                    activation = self.activation)(point_in)

            for _ in range(self.layer_num - 1):
                x = Dense(
                    units = self.hidden_size,
                    activation = self.activation)(x)

        color_out = Dense(
            units = 3,
            activation = 'sigmoid')(x)

        if self.use_sdf:
            distance_out = Dense(
                units = 1,
                activation = 'tanh')(x)
            self.model = Model(inputs=point_in, outputs=[color_out, distance_out])
        else:
            self.model = Model(inputs=point_in, outputs=color_out)

        print(self.model.summary())

    def train(self, epoch_num, point_train, color_train, distance_train,
        point_validate, color_gt, validate):
        self.model.compile(
            optimizer = Adam(lr=self.learning_rate),
            loss = MeanAbsoluteError())

        loss_color_list = []
        loss_distance_list = []
        ae_color_list = []
        mae_color_list = []

        batch_num = point_train.shape[0] // self.batch_size
        for e in range(1, epoch_num + 1):
            print('Epoch %d' % e)
            loss_color = 0
            loss_distance = 0
            for i in tqdm(range(batch_num)):
                point_batch = point_train[i * self.batch_size: (i + 1) * self.batch_size]
                color_batch = color_train[i * self.batch_size: (i + 1) * self.batch_size]
                distance_batch = distance_train[i * self.batch_size: (i + 1) * self.batch_size]
                if self.use_sdf:
                    losses = self.model.train_on_batch(point_batch, [color_batch, distance_batch])
                    loss_color += losses[1]
                    loss_distance += losses[2]
                else:
                    loss_color += self.model.train_on_batch(point_batch, color_batch)
            loss_color /= batch_num
            loss_color_list.append(loss_color)
            print('Color loss: %f' % loss_color)
            if self.use_sdf:
                loss_distance /= batch_num
                loss_distance_list.append(loss_distance)
                print('Distance loss: %f' % loss_distance)

            if validate:
                if e%10 == 0:
                    surface_color_predicted = self.model.predict(point_validate)
                    ae_color = np.absolute(np.subtract(color_gt, surface_color_predicted))
                    ae_color_list.append(ae_color)
                    mae_color = ae_color.mean()
                    mae_color_list.append(mae_color)
                    print('Color validation MAE: %f' % mae_color)

                    write_ply(os.path.join(RESULTS_COLOR_DIR, self.model_name + '_pred_' + str(e) + '.ply'), point_validate, surface_color_predicted * 255)

        return loss_color_list, loss_distance_list, ae_color_list, mae_color_list
