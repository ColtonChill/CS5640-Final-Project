
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
        # print("\tProfit:", agent.netWorth())
    print("Net Profit:", net_profit/length, agent.exploration_rate)

def train(env, agent, length=100):
    total_reward = 0
    for _ in range(length):
        done = False
        next_state = env.startEpoch()
        rewards = []
        while not done:
            action = agent.act(next_state)
            next_state, reward, done = env.step(action, agent)
            agent.learn(next_state, action, reward)
            rewards.append(reward)
        total_reward += sum(rewards)/len(rewards)/length
    return total_reward
        
def main():
    env = Env(hist_size=2**7)
    agent = TDn_Agent(env=env)
    training = []
    validation = []
    i = 1

    while(True):
        validation.append(evaluate(env,agent,length=100))
        training.append(train(env,agent))         
        if i%100==0:
            graph_eval(training, validation)
        i+=1


main()