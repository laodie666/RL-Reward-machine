import Game
import Players
import copy
import time
import matplotlib.pyplot as plt

map = Game.Two_coffee_map

game = Game.Game(map)


while not game.win:
    game.display()
    action = input()
    if action == "w":
        reward = game.step(Game.UP_COMMAND)
    
    if action == "s":
        reward = game.step(Game.DOWN_COMMAND)
        
    if action == "a":
        reward = game.step(Game.LEFT_COMMAND)
        
    if action == "d":
        reward = game.step(Game.RIGHT_COMMAND)

    print(reward)

game.display()