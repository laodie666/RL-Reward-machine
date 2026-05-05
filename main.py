import Game
import Players
import copy
import time

map = Game.More_Complicated_map

################### RANDOM PLAYER ####################

# counts = []

# for episodes in range(10):
#     step = 0
#     game = Game.Game(map)
#     Random_player = Players.RandomPlayer(game)

#     while not game.win:
#         action = Random_player.get_action(game)
#         game.step(action)
#         step += 1

#     print(f"episode {episodes} is done")
#     print(step)
#     counts.append(step)


# print(counts)

################## Q table ############################

counts = []
game = Game.Game(map)
Q_table_player = Players.QTablePlayer(game)

for episodes in range(10):
    step = 0

    while not game.win:
        action = Q_table_player.get_action(game)
        cur_game = copy.deepcopy(game)
        reward = game.step(action)
        next_game = game
        Q_table_player.update_q_value(cur_game, action, reward, next_game)
        step += 1

        # display last episode
        if episodes == 9:
            game.display()
            print()
            time.sleep(0.2)

    print(f"episode {episodes} is done")
    print(step)
    counts.append(step)
    game = Game.Game(map)

print(counts)
    
