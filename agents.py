import numpy as np
from keras.models import Sequential,load_model
import keras

#model = load_model('only_128.h5')
model = load_model('model_2048.h5')


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

class MyOwnAgent(Agent):
    def __init__(self,game,display=None):
        Agent.__init__(self, game, display)
        
    
    def step(self):
        #direction = np.random.randint(0, 4)
        
        x_to_pred = np.array(self.game.board)  #先将输入转换为numpy数组
        x_to_pred = np.log2(x_to_pred+1)
        x_to_pred = np.trunc(x_to_pred)
        x_to_pred = keras.utils.to_categorical(x_to_pred, 12) #one-hot编码
        x_to_pred = x_to_pred.reshape(1, 4, 4, 12)
        
        pred=model.predict(x_to_pred)  #这里注意模型训练的时候输入是什么
        r=pred[0]
        r1=r.tolist()
        direction=r1.index(max(r1))
        return direction
    
    
    
    
    
    
    