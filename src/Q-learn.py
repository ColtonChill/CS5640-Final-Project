
from agent import *
from enviroment import *


        
def main():
    env = Env(hist_size=2**7)
    agent = TDn_Agent(env=env)
    training = []
    validation = []
    i = 1

    state = env.startEpoch()

    graph_perfect(state['history'],label_max_mins(state['history']), env.size)


main()