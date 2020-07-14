'''
A program to implement the Sarsa on policy learning algorithm
Minimum number of steps to reach the FINISH is 15
After epsilon greedy the textbook gives an answer of 17
'''

import numpy as np
import matplotlib.pyplot as plt 
from operator import add

#DEFINING THE BOUNDARIES
ROWS = 7
COLUMNS = 10

#INITIALIZING THE START AND FINISH
START = [3,0]
FINISH = [3,7]

#SETTING THE WIND
WIND = [0,0,0,1,1,1,2,2,1,0]

#INITIALIZING THE ACTIONS
ACTIONS = [[-1,0],[+1,0],[0,-1],[0,+1]]     #UP[0], DOWN[1], LEFT[2], RIGHT[3]

#INITIALIZING Q(S,A)
qvalue = np.zeros((7,10,4))

#INITIALIZING THE HYPERPARAMETERS
REWARD = -1 
GAMMA = 1
ALPHA = 0.5
EPSILON = 0.1

#Choosing an action from a state and generating the next states
def step(pos):
    pos1 = pos
    x = pos[0]
    y = pos[1]
    action = np.argwhere(qvalue[x,y] == np.amax(qvalue[x,y]))
    action = np.concatenate(action).flat
    act = np.random.choice(action)
    move = ACTIONS[act]
    pos1 = list(map(add,pos1,move))
    x1,y1 = pos1[0],pos1[1]
    
    if(y>=3 and y <9):
        pos1[0] -= WIND[y]
    if(pos1[0]<0):
        pos1[0] += 1   
    return pos1,act


for t in range(100):
    done = False

    t = 0
    pos = START     #State S
    x = pos[0]
    y = pos[1]
    pos1,act = step(pos)    #State S'
    x1 = pos1[0]
    y1 = pos1[1]
    if(x1<0 or x1>=ROWS or y1<0 or y1>=COLUMNS):
        pos1 = pos
        x1 = pos1[0]
        y1 = pos1[1]
    while(not done):
        
        #Choose A' for S'
        pos2,act1 = step(pos1)

        #Sarsa update function 
        qvalue[x,y,act]  = qvalue[x,y,act] + ALPHA*(REWARD + GAMMA*qvalue[x1,y1,act1] - qvalue[x,y,act])

        #S<-S' and A<-A'
        x,y = pos1[0],pos1[1]
        pos1 = pos2 
        x1,y1 = pos2[0],pos2[1]
        if(x1<0 or x1>=ROWS or y1<0 or y1>=COLUMNS):
            pos1 = pos
            x1 = pos1[0]
            y1 = pos1[1]
        act = act1        
        t+=1       

        #if the episode ends
        if([x,y]==FINISH):
            done = True
print('Steps needed to reach to the GOAL is ',t)
