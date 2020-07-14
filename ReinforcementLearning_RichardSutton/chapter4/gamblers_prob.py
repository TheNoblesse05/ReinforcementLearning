'''
A program to find the optimal policy using Value Iteration.
'''

import numpy as np 

GOAL = 100
STATES = np.arange(GOAL+1)
HEAD_PROB = 0.4
REWARD = np.zeros(GOAL+1)
REWARD[GOAL] = 1
np.set_printoptions(precision=3)
MONEY = np.zeros(GOAL+1)


for t in range(15):
    for state in range(1,len(STATES)-1):
        #The possible actions you have in that state
        ACTION = np.arange(min(state,GOAL-state)+1)
        #Choosing an action except $0
        ACTION = ACTION[1:]
        winning = []
        #Calculating the reward for each possible action
        for action in ACTION:
            winning.append(HEAD_PROB*REWARD[state+action]+(1-HEAD_PROB)*REWARD[state-action])
        #Choosing the maximum reward
        REWARD[state] = np.max(winning)
        #Choosing the money that led to this maximum reward 
        x = np.argmax(np.round(winning,5))
        #Storing that action for each state
        MONEY[state] = ACTION[x]
print('The probability to win the game with the current money you have \n',REWARD*100)
print('The amount that shoud be betted in each state \n',MONEY)