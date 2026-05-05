# @ is player
# c is coffee
# # is wall
# o is office
# * is empty
# want to collect coffee then go to office.

import numpy as np
from pynput import keyboard
from pynput.keyboard import Key
import copy

UP_COMMAND = 0
DOWN_COMMAND = 1
LEFT_COMMAND = 2
RIGHT_COMMAND = 3
VALID_MOVES = [UP_COMMAND, DOWN_COMMAND, LEFT_COMMAND, RIGHT_COMMAND]

class Map:
    def __init__(self, layout, starting_pos, office_pos):
        self.layout = layout
        self.pos = starting_pos
        self.office_pos = office_pos
        self.H = len(self.layout)
        self.W = len(self.layout[0])

Simple_layout = np.array([["@", "c"], 
                          ["#", "o"]])

Simple_map = Map(Simple_layout, (0,0), (1,1))

Complex_layout =   np.array([["@", "*", "c"], 
                             ["*", "#", "*"],
                             ["o", "*", "*"]])

Complex_map = Map(Complex_layout, (0,0), (2,0))

More_Complicated_layout = np.array([["@", "#", "*", "*", "c"], 
                                    ["*", "#", "*", "#", "#"],
                                    ["*", "*", "*", "*", "o"]])

More_Complicated_map = Map(More_Complicated_layout, (0,0), (2,4))

class Game:

    def __init__(self, map = Simple_map):
        self.map = copy.deepcopy(map)
        self.has_coffee = False
        self.win = False

    def display(self):
        for row in self.map.layout:
            print(row)

    # return state, reward, done
    def step(self, dir):
        cur_pos = self.map.pos
        new_pos = list(self.map.pos)
        if dir == UP_COMMAND:
            new_pos[0] -= 1
        elif dir == DOWN_COMMAND:
            new_pos[0] += 1 
        elif dir == LEFT_COMMAND:
            new_pos[1] -= 1
        elif dir == RIGHT_COMMAND:
            new_pos[1] += 1
        else:
            print("INVALID MOVEMENT INPUT")

        new_pos = tuple(new_pos)

        if new_pos[0] < 0 or new_pos[0] >= self.map.H or new_pos[1] < 0 or new_pos[1] >= self.map.W:
            # Out of bound, nothing happens
            return -10
        elif self.map.layout[new_pos] == "#":
            # Hit a wall, nothing happens
            return -10
        else:
            reward = 0
            
            # Valid move
            if self.map.layout[new_pos] == "c":
                self.has_coffee = True
                reward = 1
            elif self.map.layout[new_pos] == "o" and self.has_coffee:
                self.win = True
                reward = 5
            else:
                reward = -0.01

            self.map.layout[cur_pos] = "*"
            if cur_pos == self.map.office_pos and self.has_coffee == False:
                # replace the office tile if coffee not delivered. 
                self.map.layout[cur_pos] = "o"
            
            self.map.layout[new_pos] = "@"
            self.map.pos = new_pos
            
            return reward
            

