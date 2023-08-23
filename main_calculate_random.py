import random

from board_env.board import Board
from board_env.player import Player
import numpy as np
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

    black_win = 0
    white_win = 0

    board = Board()

    episode = 0
    all_episode = 5000
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

        all_position = board.check_valid_position(board.player)
        if len(all_position) == 0:
            board.player = Player.get_opposite(board.player)
            continue
        position = random_get_position(all_position)
        board.move(board.player, position)

    print("黑棋是随机策略，白棋也是随机策略")
    print("黑棋胜利次数" + str(black_win))
    print("白棋胜利次数" + str(white_win))


def random_get_position(all_position):
    return random.sample(all_position, 1)[0]


play()
