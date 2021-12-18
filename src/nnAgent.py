from typing import Generator
from numpy.lib.histograms import histogram
from agent import *
from enviroment import *

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
        print("\tProfit:", agent.netWorth(),";",agent.action_history)
    print("Net Profit:", net_profit/length)

def main():
    env = Env(hist_size=2**7)
    agent = Agent(env=env)

    while(True):
        evaluate(env,agent,length=10)
        train_x,train_y = env.produceTrainingSet(batch=env.size*10)
        agent.Q.fit(x=train_x,y=train_y, validation_data=(train_x, train_y), batch_size=env.size, epochs=10)
main()