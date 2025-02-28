import gui
import snake
from kai import Network, device
import torch

model = Network(29, 3, 51).to(device)
model.load_state_dict(torch.load('./saves/snake1000.pt'))
model.eval()

game = snake.SnakeGame()

gui.init_game()
gui.update(model, game)