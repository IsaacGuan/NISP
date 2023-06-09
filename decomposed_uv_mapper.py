import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, MeanAbsoluteError
from tqdm import tqdm
from utils import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
RESULTS_DECOMPOSED_UV_DIR = os.path.join(RESULTS_DIR, 'decomposed-uv-mapper')

class DecomposedUVMapper:
    def __init__(self, model_name, components_num, fourier_max_freq=0, hidden_size=64, layer_num=8, batch_size=2048,
        activation='elu', learning_rate=0.0005, use_siren=True, use_sdf=False):
        self.model_name = model_name
        self.components_num = components_num
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

        component_out = Dense(
            units = self.components_num,
            activation = 'softmax')(x)

        self.point2component = Model(inputs=point_in, outputs=component_out)

        print(self.point2component.summary())

        component_in = Input(shape=(1,))

        if self.use_siren:
            if (self.fourier_max_freq > 0):
                x = Concatenate()(
                    [Lambda(get_fourier_features, arguments={'fourier_max_freq': self.fourier_max_freq})(point_in),
                     Lambda(get_fourier_features, arguments={'fourier_max_freq': self.fourier_max_freq, 'dim': 1})(component_in)])
                x = Dense(
                    units = self.hidden_size,
                    activation = tf.math.sin,
                    kernel_initializer = 'he_uniform',
                    bias_initializer = 'he_uniform')(x)
            else:
                x = Concatenate()([point_in, component_in])
                x = Dense(
                    units = self.hidden_size,
                    activation = tf.math.sin,
                    kernel_initializer = 'he_uniform',
                    bias_initializer = 'he_uniform')(x)

            for _ in range(self.layer_num - 1):
                x = Dense(
                    units = self.hidden_size,
                    activation = tf.math.sin,
                    kernel_initializer = 'he_uniform',
                    bias_initializer = 'he_uniform')(x)

        else:
            if (self.fourier_max_freq > 0):
                x = Concatenate()(
                    [Lambda(get_fourier_features, arguments={'fourier_max_freq': self.fourier_max_freq})(point_in),
                     Lambda(get_fourier_features, arguments={'fourier_max_freq': self.fourier_max_freq, 'dim': 1})(component_in)])
                x = Dense(
                    units = self.hidden_size,
                    activation = self.activation)(x)
            else:
                x = Concatenate()([point_in, component_in])
                x = Dense(
                    units = self.hidden_size,
                    activation = self.activation)(x)

            for _ in range(self.layer_num - 1):
                x = Dense(
                    units = self.hidden_size,
                    activation = self.activation)(x)

        uv_out = Dense(
            units = 2,
            activation = 'sigmoid')(x)

        if self.use_sdf:
            distance_out = Dense(
                units = 1,
                activation = 'tanh')(x)
            self.point2UV = Model(inputs=[point_in, component_in], outputs=[uv_out, distance_out])
            [uv_out, distance_out] = self.point2UV([point_in, tf.math.argmax(self.point2component(point_in), axis=1)])
            self.model = Model(inputs=point_in, outputs=[uv_out, distance_out])
        else:
            self.point2UV = Model(inputs=[point_in, component_in], outputs=uv_out)
            uv_out = self.point2UV([point_in, tf.math.argmax(self.point2component(point_in), axis=1)])
            self.model = Model(inputs=point_in, outputs=uv_out)

        print(self.point2UV.summary())
        print(self.model.summary())

    def train(self, epoch_num, point_train, component_train, component_onehot_train, uv_train,
        distance_train, point_validate, component_gt, uv_gt, color_gt, validate, tex):
        self.point2component.compile(
            optimizer = Adam(lr=self.learning_rate),
            loss = CategoricalCrossentropy())

        self.point2UV.compile(
            optimizer = Adam(lr=self.learning_rate),
            loss = MeanAbsoluteError())

        loss_component_list = []
        loss_uv_list = []
        loss_distance_list = []
        accuracy_component_list = []
        precision_component_list = []
        ae_uv_list = []
        mae_uv_list = []
        ae_color_list = []
        mae_color_list = []

        batch_num = point_train.shape[0] // self.batch_size
        for e in range(1, epoch_num + 1):
            print('Epoch %d' % e)
            loss_component = 0
            loss_uv = 0
            loss_distance = 0
            for i in tqdm(range(batch_num)):
                point_batch = point_train[i * self.batch_size: (i + 1) * self.batch_size]
                component_batch = component_train[i * self.batch_size: (i + 1) * self.batch_size]
                component_onehot_batch = component_onehot_train[i * self.batch_size: (i + 1) * self.batch_size]
                uv_batch = uv_train[i * self.batch_size: (i + 1) * self.batch_size]
                distance_batch = distance_train[i * self.batch_size: (i + 1) * self.batch_size]
                loss_component += self.point2component.train_on_batch(point_batch, component_onehot_batch)
                if self.use_sdf:
                    losses = self.point2UV.train_on_batch([point_batch, component_batch], [uv_batch, distance_batch])
                    loss_uv += losses[1]
                    loss_distance += losses[2]
                else:
                    loss_uv += self.point2UV.train_on_batch([point_batch, component_batch], uv_batch)
            loss_component /= batch_num
            loss_component_list.append(loss_component)
            print('Component loss: %f' % loss_component)
            loss_uv /= batch_num
            loss_uv_list.append(loss_uv)
            print('UV loss: %f' % loss_uv)
            if self.use_sdf:
                loss_distance /= batch_num
                loss_distance_list.append(loss_distance)
                print('Distance loss: %f' % loss_distance)

            if validate:
                if e%10 == 0:
                    surface_component_predicted = self.point2component.predict(point_validate)
                    surface_component_predicted = np.argmax(surface_component_predicted, axis=1)

                    accuracy, precision = compute_accuracy_precision(surface_component_predicted, component_gt)
                    accuracy_component_list.append(accuracy)
                    precision_component_list.append(precision)
                    print('Component validation accuracy: %f' % accuracy)
                    print('Component validation precision: %f' % precision)

                    surface_uv_predicted = self.model.predict(point_validate)
                    surface_color_predicted = uv_to_color(surface_uv_predicted, tex)
                    ae_uv = np.absolute(np.subtract(uv_gt, surface_uv_predicted))
                    ae_uv_list.append(ae_uv)
                    mae_uv = ae_uv.mean()
                    mae_uv_list.append(mae_uv)
                    ae_color = np.absolute(np.subtract(color_gt, surface_color_predicted))
                    ae_color_list.append(ae_color)
                    mae_color = ae_color.mean()
                    mae_color_list.append(mae_color)
                    print('UV validation MAE: %f' % mae_uv)
                    print('Color validation MAE: %f' % mae_color)

                    write_ply(os.path.join(RESULTS_DECOMPOSED_UV_DIR, self.model_name + '_pred_' + str(e) + '.ply'), point_validate, surface_color_predicted * 255)

                    plt.figure()
                    plt.axis('off')
                    plt.scatter(surface_uv_predicted[:,0], surface_uv_predicted[:,1], s=1)
                    plt.savefig(os.path.join(RESULTS_DECOMPOSED_UV_DIR, self.model_name + '_uv_layout_pred_' + str(e) + '.png'))

        return loss_component_list, loss_uv_list, loss_distance_list, accuracy_component_list, precision_component_list, ae_uv_list, mae_uv_list, ae_color_list, mae_color_list
