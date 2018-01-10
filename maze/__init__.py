from gym.envs.registration import registry, register, make, spec
from .maze import MazeEnv 

register(
    id='Maze-v0',
    entry_point='maze.maze:MazeEnv'
)