from os import stat
from random import random

from numpy.core.fromnumeric import shape
from utils import *
import tensorflow as tf


class Agent:
    def __init__(self, env) -> None:
        self.env = env
        self.action_space = 2 # 0=sell/null 1=buy/null
        self.clear()

        self.exploration_rate=0.9
        self.discount = 0.80
        self.decay_factor = 0.000000001
        self.learning_rate=0.01
        self.memories = [] # diffrent for each agent

    def clear(self):
        self.action_history = []
        self.bought_price = 0
        self.balance = 0
        self.holding = False  # holding is for the environment to toggle, not the agent
        

    def act(self, action, state):
        # These are the redundant cases, do nothing.
        if (self.holding and action==1) or (not self.holding and action == 0):
            ...
        elif action == 1: # bought 
            self.bought_price = state['currentPrice']
            self.balance -= state['currentPrice']
            self.action_history.append({'time':state['time'],'act':action})
        else: # sell
            self.balance += state['currentPrice']
            self.action_history.append({'time':state['time'],'act':action})

        return action

    def netWorth(self):
        if self.holding:
            return self.balance + self.env.currentPrice()
        else:
            return self.balance

class RandomAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
    
    def act(self, state):
        action = round(random())
        return super._act(action, state)

class QAgent(Agent):
    def __init__(self, env, n=5):
        super().__init__(env)
    
    def act(self, state):
        action = round(random())
        return super._act(action, state)

class TDn_Agent(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.bin_size = 1/250

        self.Q = self._build_table()

    def _bin_price(self, state):
        offset = round(1/self.bin_size*2)
        delta_price = state['currentPrice'] - state['lastPrice']
        index = round(delta_price/self.bin_size) + offset
        return index

    def learn(self, state, action, reward):
        if(self.exploration_rate>0):
            self.exploration_rate -= self.decay_factor
        # self.td_update(state, next_state, action, reward, next_action)
        delta_price = self._bin_price(state)
        self.memories.append((delta_price, action))

        if(reward > 0): # ie, I sold it, start distributing reward
            for ex, memory in enumerate(self.memories):
                discount_reward = (self.discount**ex) ** reward
                update = self.learning_rate*(discount_reward + self.Q[memory[0]][memory[1]])
                self.Q[memory[0]][memory[1]] += update
            self.memories.clear()
            
    def act(self, state):
        if self.exploration_rate > random():
            action = round(random())
        else:
            entry = self.Q[self._bin_price(state)]
            action = 1 if entry[1]>entry[0] else 0
        return super().act(action, state)

    def _build_table(self):
        """
        returns dict{ state : (action, reward)}
        returns dict{ delta_price_bin : {sell: reward, buy: reward}}
        """
        return [
            {'r': i, 0:0, 1:0} for i in np.arange(-2, 2, self.bin_size)
        ]

class nnAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.Q = self._build_model(env.size)

    def act(self, state):
        action = round(self.Q.predict(state["history"].reshape(1,self.env.size))[0,0])
        return super._act(action, state)


    def _build_model(self, input_size):
        """
        This input size is the size of the price history, with zeros for undiscovered histories.
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=input_size),
            tf.keras.layers.Dense(2**9, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(2**8, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(2**7, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(2**6, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(2**5, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        model.summary()
        return model