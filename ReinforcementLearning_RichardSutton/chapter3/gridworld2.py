'''
Using the Bellman Optimality equation to find the optimal policy to the girdworld
'''

import numpy as np 
from matplotlib import pyplot as plt 
import seaborn as sns


N = 5
discount = 0.9
ACTIONS = np.array([[+1,0],[-1,0],[0,+1],[0,-1]])
probability = 0.25      #for the first episode probability remains 0.25 for equi-probable action then is changed to 1

#creating girdworld
arr = np.zeros((N,N))


def calc_v(i,j):
    state = np.array([i,j])
    reward = 0
    find_max = []

    for x in range(len(ACTIONS)):
        r = 0
        if(i==0 and j==1):
            reward+=10
            a = 4
            b = 1
        elif(i==0 and j==3):
            reward+=5
            a = 2
            b = 3
        else:    
            next_state = state + ACTIONS[x]    
            a,b = next_state    
            if(a<0 or a>=N or b<0 or b>=N):        
                r = -1        
                next_state = state        
            a,b = next_state
        if(probability!=1):
            reward1 = r + discount*arr[a,b]
            reward = reward + reward1
        else:
            find_max.append(arr[a,b])
    if(probability==1):
        reward1 = discount*(max(find_max))
        reward = reward + reward1
    return (reward*probability)


def main():
    global probability
    for t in range(50):
        for i in range(N):
            for j in range(N):
                v = calc_v(i,j)
                arr[i][j] = v
        probability = 1     #probability is changed to 1 because after the first episode we already have a policy

    np.set_printoptions(precision=2)        #sets decimal places to 2
    print(arr)

main()