'''
A program to implement the Tabular Dyna-Q algorithm and depict the advantage of planning with a model.
PLANNING = 5 STEPS
'''


import numpy as np 
import matplotlib.pyplot as plt 
from operator import add
from tqdm import tqdm


#INITIALIZING THE ENVIRONMENT
ROWS = 6
COLUMNS = 9

START = [2,0] 
FINISH = [0,8]

ACTIONS = [[-1,0],[+1,0],[0,+1],[0,-1]]     #UP(0), DOWN(1), RIGHT(2), LEFT(3)
BLOCKS = [[1,2],[2,2],[3,2],[4,5],[0,7],[1,7],[2,7]]

#CREATING THE MODEL
M_STATES = []
M_ACTIONS = np.zeros((ROWS,COLUMNS,4))

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
flag = 0
for t in range(60):
    c=0
    done = False
    pos = START
    M_STATES.append(pos)
    x,y = pos[0],pos[1]
    while(not done):
        #Choose A for S
        pos1,act = step(pos)
        x1,y1 = pos1[0],pos1[1]

        #Creating the MODEL
        if(pos1 not in M_STATES):
            M_STATES.append(pos1)
        M_ACTIONS[x,y,act] = 1

        #Argmax Q(S',a)
        qmax = np.amax(qvalue[x1,y1,:])

        if(pos1 == FINISH):
            reward = 1
            done = True
        else:
            reward = 0
        #Q-Learning update function
        qvalue[x,y,act] = qvalue[x,y,act] + ALPHA*(reward + GAMMA*qmax - qvalue[x,y,act])
        #MODEL
        if(flag==1):
            for g in range(5):
                b = np.random.randint(0,len(M_STATES))
                mpos = M_STATES[b]
                if(mpos == FINISH):
                    continue
                action = M_ACTIONS[mpos[0],mpos[1],:]
                action = np.argwhere(action>0)
                action = np.concatenate(action).flat
                mact = np.random.choice(action)
                move = ACTIONS[mact]
                mpos1 = list(map(add,mpos,move))
                if(mpos1[0]<0 or mpos1[0]>=ROWS or mpos1[1]<0 or mpos1[1]>=COLUMNS):
                    mpos1 = mpos 
                if( mpos1 in BLOCKS):
                    mpos1 = mpos
                mx,my = mpos[0],mpos[1]
                mx1,my1 = mpos1[0],mpos1[1]
                qmax = np.amax(qvalue[mx1,my1,:])

                if(mpos1 == FINISH):
                    reward = 1
                else:
                    reward = 0
                qvalue[mx,my,act] = qvalue[mx,my,act] + ALPHA*(reward + GAMMA*qmax - qvalue[mx,my,act])


        pos = pos1 
        x,y = pos[0],pos[1]
        c += 1
    cc.append(c)
    flag = 1

plt.plot(cc)
plt.show()
