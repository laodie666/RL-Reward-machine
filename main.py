import Game
import Players
import copy
import time
import matplotlib.pyplot as plt

map = Game.Maze_map
episode_number = 200

################### RANDOM PLAYER ####################

# Random_counts = []

# for episodes in range(episode_number):
#     step = 0
#     game = Game.Game(map)
#     Random_player = Players.RandomPlayer(game)

#     while not game.win:
#         action = Random_player.get_action(game)
#         game.step(action)
#         step += 1

#     Random_counts.append(step)


# print(Random_counts)

################## Baseline Q table ############################
# Baseline_Q_counts = []
# game = Game.Game(map)
# Q_table_player = Players.QTablePlayer(game)

# for episodes in range(episode_number):
#     step = 0

#     while not game.win:
#         action = Q_table_player.get_action(game)
#         cur_game = copy.deepcopy(game)
#         reward = game.step(action)
#         next_game = game
#         Q_table_player.update_q_value(cur_game, action, reward, next_game)
#         step += 1

#     Baseline_Q_counts.append(step)
#     game = Game.Game(map)

# print(Baseline_Q_counts)

################## Cross Product Q table ############################

Cross_Product_Q_counts = []
game = Game.Game(map)
Q_table_player = Players.CrossProductQTablePlayer(game)

for episodes in range(episode_number):
    step = 0

    while not game.win:

        action = Q_table_player.get_action(game)
        cur_game = copy.deepcopy(game)
        reward = game.step(action)
        next_game = game
        Q_table_player.update_q_value(cur_game, action, reward, next_game)
        step += 1

    Cross_Product_Q_counts.append(step)
    game = Game.Game(map)

print(Cross_Product_Q_counts)
    
################## CRM Q table ############################

CRM_Q_counts = []
game = Game.Game(map)
Q_table_player = Players.CrossProductQTablePlayer(game)

for episodes in range(episode_number):
    step = 0

    while not game.win:
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
        
        game.step(action)
        step += 1

        # if episodes == episode_number - 1:
        #     game.display()
        #     time.sleep(0.1)

    CRM_Q_counts.append(step)
    game = Game.Game(map)

print(CRM_Q_counts)
    
x = list(range(episode_number))
# plt.plot(x, Random_counts, label = "random")
# plt.plot(x, Baseline_Q_counts, label = "baseline")
plt.plot(x, Cross_Product_Q_counts, label = "cross product")
plt.plot(x, CRM_Q_counts, label = "CRM")

plt.xlabel("X-axis data")
plt.ylabel("Y-axis data")
plt.legend()
plt.title('multiple plots')
plt.show()
