from enum import Enum
import random
from collections import deque


class Direction(Enum):
    RIGHT = [0, 1]
    UP = [-1, 0]
    LEFT = [0, -1]
    DOWN = [1, 0]


class Cell(Enum):
    NONE = 0.
    BODY = 1/3
    HEAD = 2/3
    FOOD = 1.


class GameState(Enum):
    PLAYING = 0
    GAMEOVER = 1


class SnakeGame:
    def __init__(self, board_width=5, board_height=5, length=4):
        self.bodies = deque([[board_height//2, board_width//2-i] for i in range(length)])
        self.length = length
        self.w, self.h = board_width, board_height
        self.direction = Direction.RIGHT
        self.summon_food()
    
    
    def init(self, board_width=5, board_height=5, length=4):
        self.bodies = deque([[board_height//2, board_width//2-i] for i in range(length)])
        self.length = length
        self.w, self.h = board_width, board_height
        self.direction = Direction.RIGHT
        self.summon_food()
    
    
    def turn(self, direction):
        self.direction = direction
    
    
    def summon_food(self):
        while True:
            pos = [random.randint(0, self.h-1), random.randint(0, self.w-1)]
            is_available = True
            
            for body in self.bodies:
                if pos[0] == body[0] and pos[1] == body[1]:
                    is_available = False
                    break
            
            if is_available:
                self.food = pos
                break
    
    
    def update(self):
        if [self.bodies[0][0]+self.direction.value[0], self.bodies[0][1]+self.direction.value[1]] in self.bodies:
            return GameState.GAMEOVER
        if not(0 <= self.bodies[0][0]+self.direction.value[0] < self.h and 0 <= self.bodies[0][1]+self.direction.value[1] < self.w):
            return GameState.GAMEOVER
        
        if self.bodies[0][0] == self.food[0] and self.bodies[0][1] == self.food[1]:
            self.length += 1
            self.bodies.appendleft([self.bodies[0][0]+self.direction.value[0], self.bodies[0][1]+self.direction.value[1]])
            self.summon_food()
        else:    
            self.bodies.pop()
            self.bodies.appendleft([self.bodies[0][0]+self.direction.value[0], self.bodies[0][1]+self.direction.value[1]])
        
        return GameState.PLAYING

    
    def board(self):
        res = [[Cell.NONE.value for _ in range(self.w)] for _ in range(self.h)]
        for body in self.bodies:
            res[body[0]][body[1]] = Cell.BODY.value
        res[self.bodies[0][0]][self.bodies[0][1]] = Cell.HEAD.value
        res[self.food[0]][self.food[1]] = Cell.FOOD.value
        
        return res
    
    
    def is_collision(self, dir):
        pt = self.bodies[0]
        if not(0 <= pt[0]+dir.value[0] < self.h and 0 <= pt[1]+dir.value[1] < self.w):
            return True
        if [pt[0]+dir.value[0], pt[1]+dir.value[1]] in self.bodies:
            return True
        return False
    
    
    def get_state(self):
        # dir_l = self.direction == Direction.LEFT
        # dir_r = self.direction == Direction.RIGHT
        # dir_d = self.direction == Direction.DOWN
        # dir_u = self.direction == Direction.UP
        
        # res = [
        #     (dir_u and self.is_collision(Direction.UP))or
        #     (dir_d and self.is_collision(Direction.DOWN))or
        #     (dir_l and self.is_collision(Direction.LEFT))or
        #     (dir_r and self.is_collision(Direction.RIGHT)),
            
        #     (dir_u and self.is_collision(Direction.RIGHT))or
        #     (dir_d and self.is_collision(Direction.LEFT))or
        #     (dir_u and self.is_collision(Direction.UP))or
        #     (dir_d and self.is_collision(Direction.DOWN)),

        #     (dir_u and self.is_collision(Direction.RIGHT))or
        #     (dir_d and self.is_collision(Direction.LEFT))or
        #     (dir_r and self.is_collision(Direction.UP))or
        #     (dir_l and self.is_collision(Direction.DOWN)),
            
        #     dir_l,
        #     dir_r,
        #     dir_u,
        #     dir_d,
            
        #     self.food[0] / self.h,
        #     self.food[1] / self.w,
        #     self.bodies[0][0] / self.h,
        #     self.bodies[0][1] / self.w
        # ]
        
        # return res
        
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_d = self.direction == Direction.DOWN
        dir_u = self.direction == Direction.UP
        
        res = []
        b = self.board()
        for i in range(self.h):
            for j in range(self.w):
                res.append(b[i][j])
        
        res.append(dir_l)
        res.append(dir_r)
        res.append(dir_d)
        res.append(dir_u)
        
        return res
    
    
    def step(self, action):
        if action == 0:
            if self.direction == Direction.RIGHT:
                self.direction = Direction.UP
            elif self.direction == Direction.UP:
                self.direction = Direction.LEFT
            elif self.direction == Direction.LEFT:
                self.direction = Direction.DOWN
            elif self.direction == Direction.DOWN:
                self.direction = Direction.RIGHT
        elif action == 1:
            pass
        elif action == 2:
            if self.direction == Direction.RIGHT:
                self.direction = Direction.DOWN
            elif self.direction == Direction.DOWN:
                self.direction = Direction.LEFT
            elif self.direction == Direction.LEFT:
                self.direction = Direction.UP
            elif self.direction == Direction.UP:
                self.direction = Direction.RIGHT
        
        l = self.length
        game_state = self.update()
        is_eat = self.length > l
        
        if game_state == GameState.GAMEOVER:
            terminated = True
            reward = -1.
            observation = self.get_state()
        else:
            reward = 0.
            terminated = False
            if is_eat:
                reward = 1.
            observation = self.get_state()
        
        return (observation, reward, terminated, None, None)
        
        
