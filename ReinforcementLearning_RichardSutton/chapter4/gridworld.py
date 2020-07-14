'''
A program to find the optimal policy of a GRIDWORLD using Iterative Policy Evaluation.
'''

import numpy as np 
from matplotlib import pyplot as plt 
import pdb

N = 4
arr = np.zeros((N,N))
ACTION = [[+1,0],[-1,0],[0,+1],[0,-1]]
ACTIONS = np.array(ACTION)
PROBABILITY = 0.25


def evaluate(i,j):
    reward = 0
    state = np.array([i,j])
    vals = np.arange(0)
    r = 0
    if((i==0 and j==0) or (i==N-1 and j==N-1)):
        return 0
    for action in ACTIONS:        
        newstate = state + action
        r += -1
        a,b = newstate        
        if(a<0 or a>=N or b<0 or b>=N):
            newstate = state
            a,b = newstate
        vals = np.append(vals,arr[a][b])
    rew = r + np.amax(vals)
    return (rew*PROBABILITY)
             

def main():
    
    for t in range(50):
        for i in range(N): 
            for j in range(N):
                v = evaluate(i,j)
                arr[i][j] = v
    print(arr)

main()