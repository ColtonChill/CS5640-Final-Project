from datetime import time
from agent import QAgent
from utils import *



class Env:
    """
    Stockmarket
    """
    def __init__(self, hist_size=128, fileName='data/gemini_BTCUSD_day.csv'):
        self.df = load(fileName)
        self.master_data = self.df.iloc[:,3][::-1].values  # take the opening data,
        self.size = hist_size
        self.time_index = 0

    def startEpoch(self):
        """
        The start of an epoch resets the time period and random slice of price history
        returns the starting "next_state" for the agent
        """
        self.time_index = 0
        hist_index = np.random.randint(0,len(self.master_data)-self.size*2)

        self.prices = self.__genHistSlice(hist_index, self.size*2)

        state = self.__draftState('null')
        return state

    def step(self, act, agent):
        """
        observation, reward, done, info
        """
        self.time_index += 1

        # calc reward
        if act == 0 and agent.holding:  # sold
            #? Two version of this I can use. (Using number one now)
            #?      1. current - agent.bought_price
            #?      2. overall price
            reward = self.prices[self.time_index+self.size] - agent.bought_price
            next_state = self.__draftState('sold')
        else: # bought or redundant actions
            reward = 0
            if act == 1 and not agent.holding:
                next_state = self.__draftState('bought')
            else:
                next_state = self.__draftState('null')

        # update agent holding status
        agent.holding = bool(act)

        done = (self.time_index >= (self.size -1))

        return next_state, reward, done  # next_state, reward, done

    def __draftState(self, act):
        """
        Produces the "experienced" price history of the agent
        """
        state = {
            "history": self.prices[self.time_index : self.time_index+self.size],
            "currentPrice": self.prices[self.time_index+self.size],
            "lastPrice": self.prices[self.time_index-1+self.size],
            'time':self.time_index,
            'stock_action': act
        }
        return state

    def produceTrainingSet(self, batch=4096):
        data = np.zeros((batch,self.size+1))
        for i in range(self.size):
            time_index = np.random.randint(0,len(self.master_data)-self.size-2)  # -1 for final data point
            data[i] = self.__genHistSlice(time_index, self.size+1) # +1 for the final data point
        truth = np.array(list(map(label_max_mins, data[:,-2:])))
        return data[:,:-1], truth  # x,y

    def __genHistSlice(self, hist_index, slice_size):
        randomSlice = self.master_data[hist_index : hist_index+slice_size]
        return self.__normalize(randomSlice)

    def __normalize(self,data):
        return (data-data.min())/(data.max()-data.min())
    
    def currentPrice(self):
        return self.prices[self.time_index]