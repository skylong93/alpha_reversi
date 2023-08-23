import random

from board_env.board import Board
from board_env.player import Player
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from exception.base_exception import ReversiBaseException
import os.path

"""
玩家游玩，直接加载模型，用于预测

1、先找模型是否存在，加载模型。
2、玩家选择黑棋，白棋。玩家选择黑棋，则电脑使用白棋，使用白棋的模型进行预测。
3、每次预测的时候，输入当前的局面状态，让然后会输出每个位置的价值。将当前局面可走的棋，计算出可走棋中的最大价值。就是电脑的走子点
4、然后又轮到玩家下棋，循环往复，至到终局
"""


def play():
    # 获取模型
    if os.path.exists('model_black.h5'):
        model_black = load_model('model_black.h5')
    else:
        raise ReversiBaseException("模型不存在")

    if os.path.exists('model_white.h5'):
        model_white = load_model('model_white.h5')
    else:
        raise ReversiBaseException("模型不存在")

    black_win = 0
    white_win = 0

    color = "1"
    human_player = Player.BLACK if color == str(Player.BLACK.value) else Player.WHITE
    machine_player = Player.get_opposite(human_player)
    machine_model = model_black if machine_player == Player.BLACK else model_white
    board = Board()

    episode = 0
    all_episode = 1000
    while True:
        # color = random.choice([1, 2])
        terminal, player_win = board.is_terminal()
        if terminal:
            # board.print_board()
            if player_win == Player.BLACK:
                black_win += 1
                episode += 1
                print("回合：" + str(episode) + " 胜利方：黑棋")
            else:
                white_win += 1
                episode += 1
                print("回合：" + str(episode) + " 胜利方：白棋")

            if episode >= all_episode:
                break
            board.reset()

        if board.player == human_player:
            # board.print_board()
            all_position = board.check_valid_position(board.player)
            if len(all_position) == 0:
                board.player = Player.get_opposite(board.player)
                continue
            position = random_get_position(all_position)
            board.move(board.player, position)
            continue
        else:
            pred = machine_model.predict(np.array([board.get_board()]))
            all_position = list(board.check_valid_position(board.player))
            if len(all_position) == 0:
                board.player = Player.get_opposite(board.player)
                continue
            all_pred_value = np.array([(pred[0][i[0] * 8 + i[1]]) for i in all_position])
            max_pred_index = np.argmax(all_pred_value)
            max_pred_position = all_position[max_pred_index]
            board.move(board.player, max_pred_position)
            continue

    print("黑棋是模型策略，白棋是随机策略")
    print("黑棋胜利次数" + str(black_win))
    print("白棋胜利次数" + str(white_win))


def random_get_position(all_position):
    return random.sample(all_position, 1)[0]


play()
