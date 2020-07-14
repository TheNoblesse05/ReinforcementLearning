'''
A program to implement the Tabular Dyna-Q algorithm and depict the advantage of planning with a model.
PLANNING = 0 STEPS
'''


import numpy as np 
import matplotlib.pyplot as plt 
from operator import add
from tqdm import tqdm

'''
Minimum steps to reach the goal = 14
Average over 500 iterations = 16.144
'''

#INITIALIZING THE ENVIRONMENT
ROWS = 6
COLUMNS = 9

START = [2,0] 
FINISH = [0,8]

ACTIONS = [[-1,0],[+1,0],[0,+1],[0,-1]]     #UP(0), DOWN(1), RIGHT(2), LEFT(3)
BLOCKS = [[1,2],[2,2],[3,2],[4,5],[0,7],[1,7],[2,7]]

#INITIALIZING THE HYPERPARAMETERS 
ALPHA = 0.1
EPSILON = 0.1
GAMMA = 0.95

#INITIALIZING THE Q(S,A) for all states and actions
qvalue = np.zeros((ROWS,COLUMNS,4))

#Computes the action to be taken in a state based on epsilon greedy algorithm
def step(pos):
    x,y = pos[0],pos[1]
    if(np.random.rand()>EPSILON):
        action = np.argwhere(qvalue[x,y,:] == np.amax(qvalue[x,y,:]))
        action = np.concatenate(action).flat
        act = np.random.choice(action)
    else:
        act = np.random.randint(0,4)
    move = ACTIONS[act]
    pos1 = list(map(add,pos,move))
    if(pos1[0]<0 or pos1[0]>=ROWS or pos1[1]<0 or pos1[1]>=COLUMNS):
        pos1 = pos 
    if( pos1 in BLOCKS):
        pos1 = pos
    return pos1, act

cc = []
c = 0
for t in range(60):
    c=0
    done = False
    pos = START
    x,y = pos[0],pos[1]
    while(not done):
        #Choose A for S
        pos1,act = step(pos)
        x1,y1 = pos1[0],pos1[1]
        #Argmax Q(S',a)
        qmax = np.amax(qvalue[x1,y1,:])

        if(pos1 == FINISH):
            reward = 1
            done = True
        else:
            reward = 0
        #Q-Learning update function
        qvalue[x,y,act] = qvalue[x,y,act] + ALPHA*(reward + GAMMA*qmax - qvalue[x,y,act])
        pos = pos1 
        x,y = pos[0],pos[1]
        c += 1
    cc.append(c)

plt.plot(cc)
plt.show()
