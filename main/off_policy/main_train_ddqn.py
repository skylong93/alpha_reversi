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


# board的状态不一定需要是初始状态，可以是中间任意一个状态
# experience_queue中的均是有效的位置
def collect_experience(board, model_black, model_white, episode, experience_queue):
    player = board.player
    terminal, _ = board.is_terminal()
    if terminal:
        board.reset()
    # 当前无法落子，则轮到对方
    if len(board.check_valid_position(player)) == 0:
        player = Player.get_opposite(player)
    state = board.get_board()
    player_valid_position = board.check_valid_position(player)

    epsilon = max(0.5 * (1 / (episode + 1)), 0.01)
    # 以 1-epsilon的概率，使用Q函数预测的动作，以epsilon的概率，随机选取动作
    if epsilon <= np.random.uniform(0, 1):
        if player == Player.BLACK:
            next_action_probability = model_black.predict(np.array([state]))
            next_action = divmod(np.argmax(next_action_probability[0]), 8)
        else:
            next_action_probability = model_white.predict(np.array([state]))
            next_action = divmod(np.argmax(next_action_probability[0]), 8)
    else:
        next_action = random.sample(board.check_valid_position(player), 1)[0]

    # 若此时的动作是无效的，再找一个next_action
    if not board.check_next_move(player, next_action):
        # experience_queue.append({"st": state, "p": player.value, "at": next_action, "at_valid": False})
        next_action = random.sample(board.check_valid_position(player), 1)[0]
    board.move(player, next_action)

    terminal, playerWin = board.is_terminal()
    next_state = board.get_board()

    # 黑棋的胜利reward设置为1000
    reward = 0 if not terminal else 1000 if playerWin == Player.BLACK else 1000 if playerWin == Player.WHITE else 0
    experience_queue.append(
        {"st": state, "p": player.value, "p_valid": player_valid_position, "at": next_action, "st_1": next_state,
         "r": reward})
    # if episode == 0 and reward != 0:
    #     begin_experience.add(
    #         {"st": state, "p": player.value, "at": next_action, "at_valid": True, "s_t_1": next_state, "r": reward})

    # 轮到对方
    player = Player.get_opposite(player)
    board.player = player


"""
使用ddqn下一步训练

"""


def main():
    board = Board()

    # 获取模型

    # model_black = DQN.get_model(8)
    # model_white = DQN.get_model(8)

    if os.path.exists('../../model_black.h5'):
        model_black_predict = load_model('../../model_black.h5')
        model_black_target = load_model('../../model_black.h5')
    else:
        model_black_predict = DQN.get_model(8)
        model_black_target = DQN.get_model(8)

    if os.path.exists('../../model_white.h5'):
        model_white_predict = load_model('../../model_white.h5')
        model_white_target = load_model('../../model_white.h5')
    else:
        model_white_predict = DQN.get_model(8)
        model_white_target = DQN.get_model(8)

    # 用于经验回放
    queue_size = 100
    episodes = 2
    experience_queue = deque(maxlen=queue_size)
    train_input_black = list()
    train_input_white = list()
    train_output_black = list()
    train_output_white = list()
    gamma = 0.9

    for episode in range(episodes):
        print("episode begin:" + str(datetime.now()))
        # 采样，根据行动策略，随机进行多盘游戏，直到塞满队列
        for i in range(queue_size):
            # 收集数据
            collect_experience(board, model_black_predict, model_white_predict, episode, experience_queue)

        print("collect_finish")
        # 先用有奖励的经验训练一次，使的权重有值，且此时所有位置的Q值都是已知的。
        experience_list = np.array(experience_queue)
        np.random.shuffle(experience_list)
        # for idx, experience in np.ndenumerate(experience_list):
        for idx in range(len(experience_list)):
            experience = experience_list[idx]
            # 黑棋则训练黑棋的DQN
            if experience["p"] == Player.BLACK.value:
                model_practice = model_black_predict
                model_target = model_black_target
            else:
                model_practice = model_white_predict
                model_target = model_white_target

            reward = experience["r"]
            # player_valid_position = experience["p_valid"]
            # player_valid_position_flatten = list()
            # for item in player_valid_position:
            #     player_valid_position_flatten.append(item[0] * 8 + item[1])

            y_true = model_practice.predict(np.array([experience["st"]]))[0]
            # 不对无效位置做训练
            # for valid_position in range(64):
            #     if valid_position not in player_valid_position_flatten:
            #         y_true[valid_position] = -1

            # 只有下棋位置有值，为reward，其他位置均是0
            # y_true = np.copy(y_pred)
            # 使用最优贝尔曼方程，TD算法设定y_true
            if reward == 0:
                ut_predict_1 = model_practice.predict(np.array([experience["st_1"]]))
                ut_target_1 = model_target.predict(np.array([experience["st_1"]]))
                a_max_index = np.argmax(ut_predict_1)
                y_true[experience["at"][0] * 8 + experience["at"][1]] = reward + gamma * ut_target_1[0][a_max_index]
            elif reward != 0:
                y_true[experience["at"][0] * 8 + experience["at"][1]] = float(reward)
            else:
                y_true[experience["at"][0] * 8 + experience["at"][1]] = float(reward)

            if experience["p"] == Player.BLACK.value:
                train_input_black.append(experience["st"])
                train_output_black.append(y_true)
            else:
                train_input_white.append(experience["st"])
                train_output_white.append(y_true)

            # 500次数据，就做一次训练
            if (idx + 1) % 500 == 0:
                if len(train_input_black) != 0:
                    print("fit black model")
                    model_black_predict.fit(np.array(train_input_black), np.array(train_output_black), epochs=20,
                                            batch_size=32)
                if len(train_input_white) != 0:
                    print("fit white model")
                    model_white_predict.fit(np.array(train_input_white), np.array(train_output_white), epochs=20,
                                            batch_size=32)

                train_input_black = list()
                train_input_white = list()
                train_output_black = list()
                train_output_white = list()

        model_black_predict.save('model_black.h5')
        model_black_predict.save('model_white.h5')
        model_black_target.set_weights(model_black_predict.get_weights())
        model_black_target.set_weights(model_black_predict.get_weights())


main()
print("finish!")
