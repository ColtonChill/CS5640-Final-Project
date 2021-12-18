
from typing import Generator
from numpy.lib.histograms import histogram
from agent import *
from enviroment import *
from utils import graph_eval

def evaluate(env, agent, length=1):
    net_profit = 0
    for _ in range(length):
        agent.clear()
        next_state = env.startEpoch()
        done = False
        while not done:
            action = agent.act(next_state)
            next_state, reward, done = env.step(action, agent)
        net_profit += agent.netWorth()
        # print("\tProfit:", agent.netWorth(),";",agent.action_history)
    print("Net Profit:", net_profit/length)
    return net_profit/length

def main():
    env = Env(hist_size=2**7)
    agent = RandomAgent(env=env)
    y_axis = []
    i = 1
    while(True):
        y_axis.append(evaluate(env,agent,length=100))
        if i%100==0: graph_eval(y_axis)
        i+=1
main()