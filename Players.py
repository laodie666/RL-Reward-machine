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


    def __init__(self, game: Game.Game, learning_rate = 0.1, discount_factor = 0.95, exploration_rate = 0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        self.q_table = {}

    def get_q_value(self, game: Game.Game, action: int):
        # the game state can be represented by position of the player and whether they have coffee.
        state = game.map.pos
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
        
        state = game.map.pos

        self.q_table[(state, action)] = current_q + self.learning_rate * (reward + self.discount_factor * best_future_q - current_q)

class CrossProductQTablePlayer(Player):


    def __init__(self, game: Game.Game, learning_rate = 0.1, discount_factor = 0.9, exploration_rate = 0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        self.q_table = {}

    def get_q_value(self, game: Game.Game, action: int):
        # the game state can be represented by position of the player and whether they have coffee.
        state = (game.map.pos, game.has_coffee, game.has_mail)
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
        
    def update_q_value(self, game: Game.Game, action, reward, next_game):
        best_future_q = max([self.get_q_value(next_game, action) for action in Game.VALID_MOVES])
        current_q = self.get_q_value(game, action)
        
        state = (game.map.pos, game.has_coffee, game.has_mail)

        self.q_table[(state, action)] = current_q + self.learning_rate * (reward + self.discount_factor * best_future_q - current_q)

class CRMQTablePlayer(Player):


    def __init__(self, game: Game.Game, learning_rate = 0.1, discount_factor = 0.9, exploration_rate = 0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        self.q_table = {}

    def get_q_value(self, game: Game.Game, action: int):
        # the game state can be represented by position of the player and whether they have coffee.
        state = (game.map.pos, game.has_coffee, game.has_mail)
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
        
    def update_q_value(self, game: Game.Game, action, reward, next_game):
        best_future_q = max([self.get_q_value(next_game, action) for action in Game.VALID_MOVES])
        current_q = self.get_q_value(game, action)
        
        state = (game.map.pos, game.has_coffee, game.has_mail)

        self.q_table[(state, action)] = current_q + self.learning_rate * (reward + self.discount_factor * best_future_q - current_q)

import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # four input, x pos, y pos, has coffee, has mail
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        # four output, the four movement directions
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class DQNPlayer(Player):
    def __init__(self, game:Game.Game, discount_factor = 0.95, epsilon = 1, epsilon_decay = 0.95, epsilon_min = 0.05):
        self.H = game.map.H
        self.W = game.map.W

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device is ", self.device)
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # target net is for evaluation, policy net is training
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.discount_factor = discount_factor
        
        self.episodes = 0

    def get_action(self, game):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(Game.VALID_MOVES)
            return action
        else:
            # Not mess up gradient
            with torch.no_grad():
                state = (game.map.pos[0], game.map.pos[1], game.has_coffee, game.has_mail)

                state = torch.FloatTensor(state).to(self.device)

                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()
    
    def train(self, batch):
        
        states, actions, rewards, next_states, dones = zip(*batch)
        

        states = torch.FloatTensor(states).to(self.device)
        
        # unsqueeze is needed for index of gather below
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # policy net return value for each action, take out the one corresponding to action
        
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        target = rewards + self.discount_factor * next_q_values * (1 - dones)

        # detach for not computing gradient
        loss = nn.MSELoss()(q_values, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
