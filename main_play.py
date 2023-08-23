from board_env.board import Board
from board_env.player import Player
import numpy as np
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

    while True:
        color = input("选择你的颜色：黑选1，白选2。")
        if color != str(Player.BLACK.value) and color != str(Player.WHITE.value):
            print("请选择1或者2。")
        else:
            break

    human_player = Player.BLACK if color == str(Player.BLACK.value) else Player.WHITE
    machine_player = Player.get_opposite(human_player)
    machine_model = model_black if machine_player == Player.BLACK else model_white
    board = Board()

    while True:
        terminal, player_win = board.is_terminal()
        if terminal:
            board.print_board()
            print("游戏结束。")
            print("黑棋胜利!") if player_win == Player.BLACK else print("白棋胜利!")
            exit(0)

        if board.player == human_player:
            board.print_board()
            all_position = board.check_valid_position(board.player)
            position = get_position(all_position)
            board.move(board.player, position)
            continue
        else:
            pred = machine_model.predict(np.array([board.get_board()]))
            all_position = list(board.check_valid_position(board.player))
            all_pred_value = np.array([(pred[0][i[0] * 8 + i[1]]) for i in all_position])
            max_pred_index = np.argmax(all_pred_value)
            max_pred_position = all_position[max_pred_index]
            board.move(board.player, max_pred_position)
            continue


def get_position(all_position):
    while True:
        position = input("输入你要走的位置，如第三行第二个位置，就是32.(索引从1开始)")
        if len(position) != 2 or not position.isdigit():
            print("无效输入。请输入要走的位置。")
            continue

        position_check = (int(position[0])-1, int(position[1])-1)
        if position_check not in all_position:
            print(position + "该位置为非法位置。请输入要走的位置。")
            continue

        return position_check


play()
