import numpy as np
from helper import Action
import random
import time


class AI:
    def get_movement(self, state):
        # return 'u', 'd', 'r', 'l'
        select = ['w', 'a', 's', 'd']
        ch = select[random.randint(0, 3)]
        print(ch)
        switch = {
            'w': Action.UP,
            's': Action.DOWN,
            'a': Action.LEFT,
            'd': Action.RIGHT
        }
        return switch[ch]
