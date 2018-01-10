from gym.envs.toy_text.discrete import DiscreteEnv
import numpy as np
from numpy.random import randint as rand
from collections import defaultdict
import sys
from gym import utils
import random

maps = {"default":[
        "2111",
        "0011",
        "1011",
        "1003"
    ]}


def _random_maze(width, height, complexity=.75, density=.75, n_starts = 4):
    # http://en.wikipedia.org/wiki/Maze_generation_algorithm
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))
    # Build actual maze
    Z = np.zeros(shape, dtype=bool)
    # Fill borders
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1
    # Make aisles
    for i in range(density):
        x, y = rand(0, shape[1] // 2) * 2, rand(0, shape[0] // 2) * 2
        Z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:             neighbours.append((y, x - 2))
            if x < shape[1] - 2:  neighbours.append((y, x + 2))
            if y > 1:             neighbours.append((y - 2, x))
            if y < shape[0] - 2:  neighbours.append((y + 2, x))
            if len(neighbours):
                y_,x_ = neighbours[rand(0, len(neighbours) - 1)]
                if Z[y_, x_] == 0:
                    Z[y_, x_] = 1
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_
                    
    #to text
    z = [['1' if val else '0' for val in row] for row in Z]
    
    free = np.vstack(np.where(~Z)).transpose()
    
    starts = np.random.permutation(free)[:n_starts+1]
    
    end = starts[-1]
    
    for start in starts:
        z[start[0]][start[1]] = '2'
    z[end[0]][end[1]] = '3'
    
    z = [''.join(row) for row in z]
    
    return z


class MazeEnv(DiscreteEnv):
    """
    Maze gym environment, mostly like FrozenLake 
    https://github.com/openai/gym/blob/522c2c532293399920743265d9bc761ed18eadb3/gym/envs/toy_text/frozen_lake.py
        
    """
    
    
    metadata = {'render.modes': ['human', 'ansi']}
    
    RANDOM_WIDTH = 20
    RANDOM_HEIGHT = 20
    
    def to_s(self,row, col):
        return row*self.ncol + col
    def from_s(self,s):
        return (s // self.ncol,s % self.ncol)
    
    def __init__(self, map_name = 'default', wall_hit_punishment = 1, show_map = False, win_reward = 100):
        self.show_map = show_map
        
        if map_name in maps:
            desc = maps[map_name]
        elif map_name == 'random':
            desc = _random_maze(MazeEnv.RANDOM_WIDTH, MazeEnv.RANDOM_HEIGHT)
        else:
            raise ValueError("Invalid map_name: {}".format(map_name))
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        
        to_s = lambda row,col: self.to_s(row,col)
        from_s = lambda s: self.from_s(s)
            
        
        nA = 4
        nS = nrow * ncol
        
        isd = np.array(desc == b'2').astype('float64').ravel()
        isd /= isd.sum()
        
        # test that grid functions are in same grid
        assert to_s(*from_s(123)) == 123
        
        #do action
        def inc(row, col, a):
            if a==0:
                col = max(col-1,0)
            elif a==1:
                row = min(row+1,nrow-1)
            elif a==2:
                col = min(col+1,ncol-1)
            elif a==3:
                row = max(row-1,0)
            return (row, col)
        P = defaultdict(lambda: {})
        for s in range(nS):
            row,col = from_s(s)
            for a in range(nA):
                
                next_pos = inc(row,col, a)
                do_reward = desc[next_pos] == b'3'
                
                P[s][a] = [(1.0, to_s(*next_pos) if desc[next_pos] != b'1' else s, int(do_reward)*win_reward if desc[next_pos] != b'1' else -wall_hit_punishment, do_reward) ]

        super(MazeEnv, self).__init__(nS, nA, P, isd)
        
    def _step(self, a):
        if self.show_map:
            s,r,d,dbg = super(MazeEnv, self)._step(a)
            return ((s, self.desc), r, d, dbg)
        else:
            return super(MazeEnv, self)._step(a)
        
    def _reset(self):
        if self.show_map:
            s = super(MazeEnv, self)._reset()
            return (s, self.desc)
        else:
            return super(MazeEnv, self)._reset()
        
    # from FrozenLake
    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")

        return outfile