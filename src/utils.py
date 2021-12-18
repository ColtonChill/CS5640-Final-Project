import warnings
warnings.filterwarnings("ignore")
import imp
from random import random
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from math import *

import os

def load(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print("Error, file ", path, "doesn't exist")
        return None

# def pad_to_size(x, size):
#     return np.concatenate((np.zeros(size-len(x)),x))

def label_max_mins(data):
    last_action = 0
    #TODO remove me!!!
    y=[] # sell=0, bought=1
    for i in range(len(data)-1):
        if last_action == 0:  # sold
            if data[i]<data[i+1]:
                last_action=1
        else:
            if data[i]>data[i+1]:
                last_action=0
        y.append(last_action)
    return np.array(y)

def massTruth(data):
    y = []
    for x in data:
        if(x[-1]>x[-2]):
            y.append(1)
        else:
            y.append(0)

def graph_perfect(data, actions, slice_size = 100): 
    trades= []
    last_action = 0
    for i,action in enumerate(actions):
        if action != last_action:
            trades.append({'time':i,'act':action})
            last_action=action
    print(trades)
    graph_trades(data,trades,slice_size)

def graph_trades(data, trades, slice_size = 100): 
    
    plt.figure(figsize=[10, 6])
    plt.title("Agent Trading Decisions over Price History\n(Prefect Trading Critic)")
    plt.xlabel("Day")
    plt.ylabel("price($)")

    plt.plot( # price histroy line
        list(range(slice_size)),
        data,
        label='Price History'
    )
    # print([trade['time'] for trade in trades if trade['act']==1])
    plt.plot(
        list(range(slice_size)),
        data,
        marker='o',
        lineWidth=0,
        color='r',
        markEvery=[trade['time'] for trade in trades if trade['act']==1],
        label='Bought'
    )

    plt.plot(
        list(range(slice_size)),
        data,
        marker='o',
        lineWidth=0,
        color='g',
        markEvery=[trade['time'] for trade in trades if trade['act']==0],
        label='Sold'
    )
    plt.legend()

    plt.show()

def graph_eval(train, val=None, title=''):
    plt.figure(figsize=[10, 6])
    plt.title(title +" Agent Profits vs. Epoch\n")
    plt.xlabel("Epoch")
    plt.ylabel("Profit (avg of 100 rounds)")

    x = np.array(list(range(len(train))))
    y = train
    m, b = np.polyfit(x, y, 1)

    plt.plot(
        x, y,
        label='Training Scoring'
    )
    if val==None:
        plt.plot(
            x, m*x+b,
            color='g',
            label="Best fit"
        )
        plt.text(1, 0.8, 'y = ' + '{:.5f}'.format(b) + ' + {:.5f}'.format(m) + 'x', size=10)
    else:
        plt.plot(
            list(range(len(val))),
            val,
            label='Validation Scoring',
            color='r',
        )

    plt.ylim([-1,1])

    plt.legend()
    plt.savefig("DTn.png")
