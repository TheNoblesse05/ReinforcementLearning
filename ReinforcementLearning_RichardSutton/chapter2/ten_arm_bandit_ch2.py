'''
A probability distribution for a ten arm testbed 
'''

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import trange

plt.style.use('ggplot')

steps = 1000
average_reward = np.zeros(10)

sam = np.random.randn(10)
sam1 = np.random.randn(1000,10)
d = sam1 + sam
print('Actual values are')
print(sam)
plt.violinplot(dataset=d)
plt.show()

show_choices = []

#Greedy
# for i in range(1000):
#     choice = np.argwhere(average_reward == np.amax(average_reward))
#     choice = choice.transpose()
#     choice = choice.ravel()
#     print('choices are ',choice)
#     my_choice = np.random.choice(choice)
#     print('my choice is ', my_choice)
#     reward = sam[my_choice] + sam1[i][my_choice]
#     #print('reward is ', reward)
#     average_reward[my_choice] = (average_reward[my_choice]*(i) + reward)/(i+1)
#     #print('average reward is ',average_reward)
#     show_choices.append(my_choice)

#Epsilon Greedy (e = 0.1)
e = 0.1
full = np.arange(10)

for i in range(1000):
    choice = np.argwhere(average_reward == np.amax(average_reward))
    choice = choice.transpose()
    choice = choice.ravel()
    if(np.random.rand()<e):
        my_choice = np.random.choice(full)
    else:
        my_choice = np.random.choice(choice)
    reward = sam[my_choice] + sam1[i][my_choice]
    average_reward[my_choice] = (average_reward[my_choice]*(i) + reward)/(i+1)
    show_choices.append(my_choice)

print('average reward = ',average_reward)
print('show choices = ',show_choices)
