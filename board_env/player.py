from enum import Enum, unique


@unique
class Player(Enum):
    BLANK = 0
    BLACK = 1
    WHITE = 2

    @staticmethod
    def get_opposite(player):
        if player == Player.BLACK:
            return Player.WHITE
        if player == Player.WHITE:
            return Player.BLACK
        return None
