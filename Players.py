import Game
import random
import numpy as np

class Player:
    def __init__(self):
        pass

    def get_action(self, game):
        pass

class RandomPlayer(Player):
    def __init__(self):
        pass

    def get_action(self, game):
        return random.choice(Game.VALID_MOVES)
  
class QTablePlayer(Player):

    def __init__(self):
        

    def get_action(self, game):
        