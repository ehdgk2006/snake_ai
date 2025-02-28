import pygame
import sys
from time import sleep
import snake
from kai import device
import torch
import numpy as np


GREEN = (10, 200, 10)
RED = (200, 10, 10)
BLACK = (0, 0, 0)
GRAY = (20, 20, 20)
screen_width = 280
screen_height = 280


def init_game():
    global screen, clock
    
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('my name is snake')
    clock = pygame.time.Clock()


def update(model, game):
    global screen, clock
    
    is_end = False
    
    while not is_end:
        for event in pygame.event.get():
            if event.type in [pygame.QUIT]:
                pygame.quit()
                sys.exit()
            if event.type in [pygame.KEYDOWN]:
                if event.key == pygame.K_RIGHT and game.direction != snake.Direction.LEFT:
                    game.turn(snake.Direction.RIGHT)
                if event.key == pygame.K_UP and game.direction != snake.Direction.DOWN:
                    game.turn(snake.Direction.UP)
                if event.key == pygame.K_LEFT and game.direction != snake.Direction.RIGHT:
                    game.turn(snake.Direction.LEFT)
                if event.key == pygame.K_DOWN and game.direction != snake.Direction.UP:
                    game.turn(snake.Direction.DOWN)
                if event.key == pygame.K_r:
                    game.init()
                    
        
        with torch.no_grad():
            z_distribution = torch.from_numpy(
                np.array([[-10 + i * 0.4 for i in range(51)]])
                )
            z_distribution = torch.unsqueeze(z_distribution, 2).float().to(device)

            Q_dist, _ = model.forward(torch.tensor(game.get_state(), dtype=torch.float, device=device))
            Q_dist = Q_dist.detach()
            Q_target = torch.matmul(Q_dist, z_distribution)

            action = Q_target.argmax(dim=1)[0].detach().cpu().numpy()[0]
        observation, reward, terminated, _, _ = game.step(action)
        
        if terminated:
            game.init()
        
        screen.fill(BLACK)
        
        if game is not None:
            for i in range(game.h):
                for j in range(game.w):
                    pygame.draw.rect(screen, GRAY, (25*j+5, 25*i+5, 20, 20))
            pygame.draw.rect(screen, RED, (25*game.food[1]+5, 25*game.food[0]+5, 20, 20))
            
            for body in game.bodies:
                pygame.draw.rect(screen, GREEN, (25*body[1]+5, 25*body[0]+5, 20, 20))
        
        pygame.display.update()
        clock.tick(10)
    pygame.quit()
