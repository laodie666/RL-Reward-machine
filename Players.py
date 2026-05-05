import Game
import random
import numpy as np

class Player:
    def __init__(self, game):
        pass

    def get_action(self, game):
        pass

class RandomPlayer(Player):
    def __init__(self, game):
        pass

    def get_action(self, game):
        return random.choice(Game.VALID_MOVES)
  
class QTablePlayer(Player):


    def __init__(self, game: Game.Game, learning_rate = 0.1, discount_factor = 0.9, exploration_rate = 0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        self.q_table = {}

    def get_q_value(self, game: Game.Game, action: int):
        # the game state can be represented by position of the player and whether they have coffee.
        state = (game.map.pos, game.has_coffee)
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0
        
        return self.q_table[(state, action)]

    def get_action(self, game: Game.Game):
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.choice(Game.VALID_MOVES)

            return action
        else:
            # return argmax action that give best q_value
            action_q_pair = [(action, self.get_q_value(game, action)) for action in Game.VALID_MOVES]
            return max(action_q_pair, key = lambda x :x[1])[0]
        
    def update_q_value(self, game, action, reward, next_game):
        best_future_q = max([self.get_q_value(next_game, action) for action in Game.VALID_MOVES])
        current_q = self.get_q_value(game, action)
        
        state = (game.map.pos, game.has_coffee)

        self.q_table[(state, action)] = current_q + self.learning_rate * (reward + self.discount_factor * best_future_q - current_q)
