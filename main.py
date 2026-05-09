import Game
import Players
import copy
import time
from collections import deque 
import random

import matplotlib.pyplot as plt

map = Game.Maze_map
episode_number = 500
step_limit = 4000
batch_size = 512

################### RANDOM PLAYER ####################

# Random_reward_per_step = []

# for episodes in range(episode_number):
#     step = 0
#     total_reward = 0
#     game = Game.Game(map)
#     Random_player = Players.RandomPlayer(game)

#     while not game.win and step < step_limit:
#         action = Random_player.get_action(game)
#         total_reward += game.step(action)
#         step += 1

#     Random_reward_per_step.append(total_reward/step)
#     print(step)

# print(Random_reward_per_step)

################## Baseline Q table ############################
# Baseline_Q_reward_per_step = []
# game = Game.Game(map)
# Q_table_player = Players.QTablePlayer(game)

# for episodes in range(episode_number):
#     step = 0
#     total_reward = 0
#     while not game.win and step < step_limit:
#         action = Q_table_player.get_action(game)
#         cur_game = copy.deepcopy(game)
#         reward = game.step(action)
#         total_reward += reward
#         next_game = game
#         Q_table_player.update_q_value(cur_game, action, reward, next_game)
#         step += 1

#     Baseline_Q_reward_per_step.append(total_reward/step)
#     game = Game.Game(map)

# print(Baseline_Q_reward_per_step)

################## Cross Product Q table ############################

Cross_Product_Q_reward_per_step = []
game = Game.Game(map)
Q_table_player = Players.CrossProductQTablePlayer(game)

for episodes in range(episode_number):
    step = 0
    total_reward = 0

    while not game.win and step < step_limit:

        action = Q_table_player.get_action(game)
        cur_game = copy.deepcopy(game)
        reward = game.step(action)
        total_reward += reward
        next_game = game
        Q_table_player.update_q_value(cur_game, action, reward, next_game)
        step += 1

    Cross_Product_Q_reward_per_step.append(total_reward/step)
    game = Game.Game(map)

print(Cross_Product_Q_reward_per_step)
    
################## CRM Q table ############################

CRM_Q_reward_per_step = []
game = Game.Game(map)
Q_table_player = Players.CRMQTablePlayer(game)

for episodes in range(episode_number):
    step = 0
    total_reward = 0
    while not game.win and step < step_limit:
        action = Q_table_player.get_action(game)
        
        # Iterate through all the reward states after step
        for has_coffee in  [False, True]:
            for has_mail in [False, True]:
                cur_game = copy.deepcopy(game)

                # simluate reward with different position in reward machine
                cur_game.has_coffee = has_coffee
                cur_game.has_mail = has_mail

                next_game = copy.deepcopy(cur_game)
                reward = next_game.step(action)
                
                Q_table_player.update_q_value(cur_game, action, reward, next_game)
        
        total_reward += game.step(action)
        step += 1

        # if episodes == episode_number - 1:
        #     game.display()
        #     time.sleep(0.1)

    CRM_Q_reward_per_step.append(total_reward/step)
    game = Game.Game(map)

print(CRM_Q_reward_per_step)



################## Cross product DQN ############################

# reward_per_step = []
# game = Game.Game(map)
# DQN_player = Players.DQNPlayer(game)
# Cross_Product_DQN_reward_per_step = []
# memory = deque(maxlen=20000)


# for episodes in range(episode_number):
#     step = 0
#     total_reward = 0
#     print(f"episode {episodes} starting")

#     while not game.win and step < step_limit:
#         action = DQN_player.get_action(game)
#         cur_game = copy.deepcopy(game)

#         reward = game.step(action)
#         total_reward += reward
#         next_game = game

#         state = (cur_game.map.pos[0], cur_game.map.pos[1], cur_game.has_coffee, cur_game.has_mail)
#         next_state = (next_game.map.pos[0], next_game.map.pos[1], next_game.has_coffee, next_game.has_mail)
#         done = game.win

#         memory.append((state, action, reward, next_state, done))

#         if len(memory) >= batch_size:
#             if len(memory) < 1000 or step % 4 == 0:
#                 batch = random.sample(memory, batch_size)
#                 DQN_player.train(batch)


#         step += 1

#         # if episodes == episode_number - 1:
#         #     game.display()
#         #     time.sleep(0.1)
#         #     print()
        
#     print(f"step {step} {game.win}")

        
#     if episodes % 5 == 0: 
#         DQN_player.update_network()

#     print(total_reward/step)
#     Cross_Product_DQN_reward_per_step.append(total_reward/step)
#     game = Game.Game(map)
    

# print(Cross_Product_DQN_reward_per_step)


################## CRM DQN ############################

# reward_per_step = []
# game = Game.Game(map)
# DQN_player = Players.DQNPlayer(game)
# CRM_DQN_reward_per_step = []
# memory = [deque(maxlen=20000) for i in range(4)]


# for episodes in range(episode_number):
#     step = 0
#     total_reward = 0
#     print(f"episode {episodes} starting")

#     while not game.win and step < step_limit:
#         action = DQN_player.get_action(game)
#         for has_coffee in  [False, True]:
#             for has_mail in [False, True]:
#                 cur_game = copy.deepcopy(game)

#                 cur_game.has_coffee = has_coffee
#                 cur_game.has_mail = has_mail
                
#                 next_game = copy.deepcopy(cur_game)
#                 reward = next_game.step(action)

#                 state = (cur_game.map.pos[0], cur_game.map.pos[1], cur_game.has_coffee, cur_game.has_mail)
#                 next_state = (next_game.map.pos[0], next_game.map.pos[1], next_game.has_coffee, next_game.has_mail)
#                 done = next_game.win

#                 memory[has_coffee + 2 * has_mail].append((state, action, reward, next_state, done))

#         reward = game.step(action)

#         total_reward += reward

        
#         for i in range(4):
#             if len(memory[i]) >= batch_size:
#                 if len(memory[i]) < 1000 or step % 4 == 0:
#                     batch = random.sample(memory[i], batch_size)
#                     DQN_player.train(batch)

#         step += 1

#         # if episodes == episode_number - 1:
#         #     game.display()
#         #     time.sleep(0.2)
#         #     print()
        
    
#     print(f"step {step} {game.win}")
        
#     if episodes % 5 == 0: 
#         DQN_player.update_network()


#     CRM_DQN_reward_per_step.append(total_reward/step)
#     print(f"reward per step {total_reward/step}")
#     game = Game.Game(map)
    

# print(CRM_DQN_reward_per_step)

#############################################################
    
x = list(range(episode_number))
# plt.plot(x, Random_reward_per_step, label = "random")
# plt.plot(x, Baseline_Q_reward_per_step, label = "baseline")
plt.plot(x, Cross_Product_Q_reward_per_step, label = "cross product")
plt.plot(x, CRM_Q_reward_per_step, label = "CRM")
# plt.plot(x, Cross_Product_DQN_reward_per_step, label = "DQN")
# plt.plot(x, CRM_DQN_reward_per_step, label = "CRM DQN")

plt.xlabel("Episode number")
plt.ylabel("Reward per step")
plt.legend()
plt.title('Training progress')
plt.show()
