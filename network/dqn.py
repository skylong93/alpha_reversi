import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque

"""
DQN的神经网络模型，采用全连接网络执行。
主要有以下几个方法

"""


class DQN(object):
    model_switch = 'simple_DQN'

    @staticmethod
    def get_model(board_size, model='optimize_DQN'):
        if model == 'optimize_DQN':
            return DQN.optimize_DQN(board_size)
        else:
            return DQN.simple_DQN(board_size)

    @staticmethod
    def simple_DQN(board_size):
        action_space = board_size * board_size
        model = Sequential()
        model.add(Dense(action_space, input_shape=(action_space,), activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    @staticmethod
    def optimize_DQN(board_size):
        action_space = board_size * board_size
        inputs = keras.Input(shape=(board_size, board_size, 1), name="board")
        x = layers.Conv2D(32, 3, activation="relu")(inputs)
        x = layers.Conv2D(32, 3, activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Flatten()(x)

        # 对决网络，需要有两个Dense网络
        v = layers.Dense(512, activation='relu')(x)
        v_outputs = layers.Dense(action_space, activation='linear')(v)

        d = layers.Dense(512, activation='relu')(x)
        d_outputs = layers.Dense(1, activation='linear')(d)

        v_mean = layers.Lambda(lambda x: tf.reduce_mean(x))(v_outputs)
        d_add = layers.Lambda(lambda x_add: x_add[0] + x_add[1])([d_outputs, v_mean])
        outputs = layers.Lambda(lambda y: tf.add(y[0], y[1]))([v_outputs, d_add])

        model = keras.Model(inputs, outputs, name="ddqn")
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        # model.summary()
        return model

DQN.optimize_DQN(8)
