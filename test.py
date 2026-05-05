import Game
import Players
import copy
import time
import matplotlib.pyplot as plt

map = Game.Simple_map

game = Game.Game(map)


while not game.win:
    game.display()
    action = input()
    if action == "w":
        game.step(Game.UP_COMMAND)
    
    if action == "s":
        game.step(Game.DOWN_COMMAND)
        
    if action == "a":
        game.step(Game.LEFT_COMMAND)
        
    if action == "d":
        game.step(Game.RIGHT_COMMAND)

game.display()