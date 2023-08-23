from board_env.board import Board
from board_env.player import Player
from network.dqn import DQN
from collections import deque
import numpy as np
import random
# from random import sample
from datetime import datetime
from tensorflow.keras.models import load_model
import os.path


def main():
    board = Board()

    # 获取模型
    model_black_predict = DQN.get_model(8)

    print(board.get_board())
    result = model_black_predict.predict(np.array([board.get_board()]))
    print('第一次预测：')
    print(result)

    train_input_black = board.get_board()
    train_output_black = board.get_board().flatten()
    for x in range(64):
        train_output_black[x] = float(-99)

    train_output_black[2 * 8 + 4] = float(1)
    train_output_black[3 * 8 + 5] = float(1)
    train_output_black[4 * 8 + 2] = float(1)
    train_output_black[5 * 8 + 3] = float(1)

    train_input = list()
    train_input.append(train_input_black)
    train_output = list()
    train_output.append(train_output_black)
    model_black_predict.fit(np.array(train_input), np.array(train_output), epochs=100, batch_size=32)
    model_black_predict.save('test_model.h5')
    result2 = model_black_predict.predict(np.array([board.get_board()]))
    print('第二次预测：')
    print(result2)


main()
