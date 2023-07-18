import numpy as np
from board_env.player import Player
from exception.base_exception import ReversiBaseException

"""
黑白棋的棋盘。
在马尔科夫决策过程中，环境主要决定了状态转移函数。也就是根据s_t,a_t,给出下一个状态s_{t+1}。
至于不同环境的动作，有多种情况。
    一种是如马里奥游戏一样，马里奥角色有着自己固定的几个动作，不受环境影响。
    另外一种，是如这个黑白棋的棋盘一样，根据不同的状态，会有不同的动作选择。

该类提供游戏环境，主要的几个方法如下：
1、根据当前状态 s_t,提供可行的多个a_t。
2、根据当前状态与动作 s_t,a_t,给出下一个状态s_{t+1}
3、判断游戏是否结束，以及谁赢了

棋盘中0表示无棋子，1表示黑棋，2表示白棋
"""


class Board(object):
    _board = None
    player = None
    _valid_position = None
    _directions = np.array([(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)])

    def __init__(self):
        self._board = np.zeros((8, 8))
        self._valid_position = {"BLACK": set(), "WHITE": set()}
        self._board[3][3] = Player.BLACK.value
        self._board[4][4] = Player.BLACK.value
        self._board[3][4] = Player.WHITE.value
        self._board[4][3] = Player.WHITE.value
        self.player = Player.BLACK

    def reset(self):
        self.__init__()
        # self._board = np.zeros((8, 8))
        # self._valid_position = {"BLACK": set(), "WHITE": set()}
        # self._board[3][3] = Player.BLACK.value
        # self._board[4][4] = Player.BLACK.value
        # self._board[3][4] = Player.WHITE.value
        # self._board[4][3] = Player.WHITE.value
        # self.player = Player.BLACK

    """
    根据当前状态s_t,提供可行的多个a_t
    
    落点的三个要求
    1、空白点
    2、（假设轮到黑子走子），落点附近的8格必须有一个白子
    3、落点的8个方向上必须存在至少一个黑子
    
    目前采取的策略是遍历棋盘中自己的棋子，看周围空白点，判断空白点是否可下，在加上位标志的方式，看哪些点被访问过了。
    """

    def check_valid_position(self, player):
        if len(self._valid_position.get(player.name)) != 0:
            return self._valid_position.get(player.name)

        for i in range(8):
            for j in range(8):
                # 如果不是自己棋子，则跳过
                if self._board[i][j] != player.value:
                    continue
                # 寻找8个方向上，是否先有对方的棋子，然后再有空白点。空白点且不在result数组中
                find_opposite = False
                for direction in self._directions:
                    x, y = i + direction[0], j + direction[1]
                    while True:
                        if x < 0 or x > 7 or y < 0 or y > 7:
                            break
                        if self._board[x][y] == player.value:
                            find_opposite = False
                        elif self._board[x][y] == Player.get_opposite(player).value:
                            find_opposite = True
                        elif self._board[x][y] == Player.BLANK.value:
                            if find_opposite:
                                self._valid_position.get(player.name).add((x, y))
                                find_opposite = False
                            break
                        else:
                            raise ReversiBaseException("错误的player枚举")
                        x, y = x + direction[0], y + direction[1]
        return self._valid_position.get(player.name)

    def check_next_move(self, player, position):
        valid_position = self.check_valid_position(player)
        if position not in valid_position:
            return False
        else:
            return True

    """
    player为Player枚举类
    position为位置的元组，如(1,3)
    """

    def move(self, player, position):
        # 检查合法性
        if not self.check_next_move(player, position):
            raise ReversiBaseException("错误落子位置，该位置非法")

        # 走子
        self._board[position[0], position[1]] = player.value
        # 8个方向上的子都翻转成player的颜色
        for direction in self._directions:
            x, y = position[0] + direction[0], position[1] + direction[1]
            reversi_flag = False
            touch_opposite = False
            while True:
                if x < 0 or x > 7 or y < 0 or y > 7:
                    break
                if self._board[x][y] == Player.get_opposite(player).value:
                    touch_opposite = True
                    # self._board[x][y] = player.value
                elif self._board[x][y] == player.value:
                    if touch_opposite:
                        reversi_flag = True
                    break
                elif self._board[x][y] == Player.BLANK.value:
                    break
                x, y = x + direction[0], y + direction[1]

            x, y = position[0] + direction[0], position[1] + direction[1]
            if reversi_flag:
                while True:
                    if x < 0 or x > 7 or y < 0 or y > 7:
                        break
                    if self._board[x][y] == Player.get_opposite(player).value:
                        self._board[x][y] = player.value
                    elif self._board[x][y] == player.value:
                        break
                    else:
                        raise ReversiBaseException("真正翻转的时候，错误的else判断位置")
                    x, y = x + direction[0], y + direction[1]

        # 清空下一步可行位置
        self.player = Player.get_opposite(self.player)
        self._valid_position = {"BLACK": set(), "WHITE": set()}
        self.check_valid_position(Player.get_opposite(player))
        return self._board

    def is_terminal(self):
        valid_black = self.check_valid_position(Player.BLACK)
        valid_white = self.check_valid_position(Player.WHITE)
        if len(valid_black) != 0 or len(valid_white) != 0:
            return False, None

        # 数子
        black = 0
        white = 0
        for i in range(8):
            for j in range(8):
                if self._board[i][j] == Player.BLACK.value:
                    black += 1
                elif self._board[i][j] == Player.WHITE.value:
                    white += 1

        if black > white:
            return True, Player.BLACK
        elif black < white:
            return True, Player.WHITE
        else:
            return True, None

    def get_board(self):
        return self._board

    def print_board(self):
        for line in self._board:
            for a in line:
                if a == 0:
                    print('·', end=' ')
                elif a == 1:
                    print('●', end=' ')
                elif a == 2:
                    print('○', end=' ')
                else:
                    print('I', end=' ')
            print('')
