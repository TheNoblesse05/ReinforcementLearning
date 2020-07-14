import numpy as np 
from matplotlib import pyplot as plt 
from tqdm import tqdm 
import pdb

#A program to calculate the approximate state-value functions for the blackjack game that sticks to only on 21 or 20.
#Program also calculates the policy the player should use.

#REWARD is +1,-1,0 for winning, losing and draw
#The states depend on Player's cards' sum (12-21), Dealer's showing card(ace-10) and usable Ace.    10x10x2 = 200 states
#Dealer's strategy is to hit until a sum of 17 or greater is reached
#WE DO NOT DISCOUNT. GAMMA = 1
#Player's actions are to Hit or Stick

STATE = np.zeros((20,10,3))

CARDS = [1,2,3,4,5,6,7,8,9,10]
CARD_SUM = [12,13,14,15,16,17,18,19,20,21]

STATE[10:,:,2] = 1
STATE[:,:,0] = CARD_SUM
for i in range(10):
    STATE[i,:,1] = STATE[i+10,:,1] =i+1


HIT = 1
STICK = 0
ACTION = [HIT,STICK]      #1 for HIT and 0 for STICK

REWARDS = np.zeros((20,10,7)) #State value, total matches, wins, loses, draws, hit, stick

print('State value\tTotal Games Wins\tLoses\t\tDraws\tHit\tStick')

#training over 500,000 eps
for t in tqdm(range(500000)):
    #dealer's policy
    cards_given = []
    op_card = np.random.choice(CARDS)
    cards_given.append(op_card)
    dealer_sum = op_card
    x=0
    c1 = 1
    while(dealer_sum<17):
        c = np.random.choice(CARDS)
        dealer_sum += c 
        c1 = c1 + 1
        cards_given.append(c)
        if(x==1 and dealer_sum>21):
            dealer_sum -=10
            x=2
        if((cards_given[0]==1 or cards_given[1]==1 or c==1) and x == 0):
            dealer_sum += 10
            x=1
            if(dealer_sum>21):
                dealer_sum -= 10
                x = 2
    

    #player's policy
    states_visited = []
    player_sum = np.random.choice(CARDS)
    states_visited.append(player_sum)
    pcards_given = []
    pcards_given.append(player_sum)
    x = 0       #ace is not with us
    c2 = 1
    y = 1
    yy = []
    while(player_sum<20 and y == 1):
        c = np.random.choice(CARDS)
        player_sum += c
        c2 = c2 + 1
        pcards_given.append(c)
        states_visited.append(player_sum)
        y = np.random.choice(ACTION)
        if(player_sum<12):
            y=1
        yy.append(y)
        if(x==1 and player_sum>21):
            player_sum -= 10
            x = 2       #ace was given but its value is taken as 1 - not usable
            states_visited.pop()
            states_visited.append(player_sum)
        if((pcards_given[0]==1 or pcards_given[1]==1 or c == 1) and x==0):
            player_sum += 10
            x = 1       #ace was given but its value is taken as 11 - usable
            states_visited.pop()
            states_visited.append(player_sum)
            if(player_sum>21):
                player_sum -= 10
                x = 2
                states_visited.pop()
                states_visited.append(player_sum)


    #implementing winning scheme
    winning = 0
    if(player_sum>21 or (dealer_sum>player_sum and dealer_sum<=21)):
        winning = -1
    elif(player_sum>dealer_sum or dealer_sum>21):
        winning = +1
    elif(player_sum==21 and c2==2 and c1>2):
        winning = +1
    elif(player_sum==21 and c1==2 and c2>2):
        winning = -1


    #implementing MC
    #putting the winning in the reward corresponding to its state
    for j,i in enumerate(states_visited):
        if(i>=12 and i<=21):
            if(x==0 or x==2): #no usable ace
                REWARDS[op_card-1,i-12,1] += 1
                g = REWARDS[op_card-1,i-12,1]
                REWARDS[op_card-1,i-12,0] = (REWARDS[op_card-1,i-12,0]*(g-1) + winning)/(g)
                if(winning==1):
                    REWARDS[op_card-1,i-12,2] += 1
                elif(winning==-1):
                    REWARDS[op_card-1,i-12,3] +=1
                else:
                    REWARDS[op_card-1,i-12,4] += 1
                REWARDS[op_card-1,i-12,6-yy[j-1]] = (REWARDS[op_card-1,i-12,6-yy[j-1]]*(g-1) + winning)/(g)
                
            else: #usable ace
                REWARDS[op_card-1+10,i-12,1] += 1
                g = REWARDS[op_card-1+10,i-12,1]
                REWARDS[op_card-1+10,i-12,0] = (REWARDS[op_card-1+10,i-12,0]*(g-1) + winning)/(g)
                if(winning==1):
                    REWARDS[op_card-1+10,i-12,2] += 1
                elif(winning==-1):
                    REWARDS[op_card-1+10,i-12,3] +=1
                else:
                    REWARDS[op_card-1+10,i-12,4] += 1
                REWARDS[op_card-1+10,i-12,6-yy[j-1]] = (REWARDS[op_card-1+10,i-12,6-yy[j-1]]*(g-1) + winning)/(g)

print(REWARDS)
print('State value\tTotal Games Wins\tLoses\t\tDraws\tHit\tStick')

for i in range(20):
    for j in range(10):
        print(STATE[i,j],end='-')
        if(REWARDS[i,j,5]>REWARDS[i,j,6]):
            print('HIT')
        else:
            print('STICK')
        