'''
A program to implement TD(0) algorithm
'''

import numpy as np 
from matplotlib import pyplot as plt

np.set_printoptions(precision=3)

#INITIALIZING THE MOVES
LEFT = -1
RIGHT = 1
ACTION = [LEFT,RIGHT]

#GENERATING THE ENVIRONMENT
arr = np.zeros(7)
arr[6] = 1


s = 3   #starting state
s1 = s

#INITIALIZING THE HYPERPARAMETERS
ALPHA = 0.1
GAMMA = 1
R = 0

for t in range(100):
    while(True):
        a = np.random.choice(ACTION)
        if(a==-1):
            s1 = s-1
        else:
            s1 = s+1
        
        arr[s] = arr[s] + ALPHA*(R + GAMMA*arr[s1] - arr[s])
        if(s1==0 or s1==6):
            s = 3
            break
        s = s1

print('State values are',arr)

brr = np.arange(7)/6 
print('Original values are',brr)

x = ['0','A','B','C','D','E','1']
plt.plot(x,arr,label='calculated1')
plt.plot(x,brr,label='actual')
plt.legend()
plt.show()

r = (arr - brr)**2
r = sum(r) /100
r = r**0.5
print('RMS ERROR FOR ALPHA 0.1 IS ',r)
