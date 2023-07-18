import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input,Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque

"""
DQN的神经网络模型，采用全连接网络执行。
主要有以下几个方法

"""


class DQN(object):

    @staticmethod
    def get_model(board_size):
        action_space = board_size * board_size

        model = Sequential()
        model.add(Dense(64, input_shape=(64,), activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model
