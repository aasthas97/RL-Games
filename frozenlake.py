"""Creating a random frozen lake instance of the given size and number of holes and using Q-learning to find the best strategy
Author: Aastha"""

import random
import numpy
from matplotlib import pyplot as plt

def MakeLake(lake_size, num_holes):
    frozen_lake = []
    for num in range(lake_size):
        frozen_lake.append([]) # add empty lists to function as rows
    for num in range(lake_size):
        while len(frozen_lake[num]) < lake_size:
            frozen_lake[num].append(0) # fill the rows with 0s

    frozen_lake[lake_size-1][lake_size-1] = 100 # last element is the goal state

    holes_generated = 0
    while holes_generated < num_holes: # generating holes
        ind1= random.randint(0, lake_size-1)
        ind2= random.randint(0, lake_size-1)
        if frozen_lake[ind1][ind2] == 100: # goal state can't be a hole
            pass
        elif ind1 == 0 and ind2 == 0: # start state can't be a hole
            pass
        elif frozen_lake[ind1][ind2] == -100: # already a hole
            pass
        else:
            frozen_lake[ind1][ind2] = -100
            holes_generated += 1

    return frozen_lake

def GetAction(state, epsilon, qtable): # e-greedy strategy
    i = random.random() # generate a random float between 0 and 1
    if i < epsilon: # explore
        action = random.choice([0, 1, 2, 3])
    else: # exploit
        action = numpy.argmax(qtable[state,:]) # get the index of the maximum q value possible in the current row
    return action

total_episodes = 500
max_steps = 500
learning_rate = 0.8 # alpha
gamma = 0.7
epsilon = 0.9
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005

lake_size = int(input("Size of lake: "))
num_holes = int(input("Number of holes: "))

env_lake = MakeLake(lake_size, num_holes)
qtable = numpy.zeros((lake_size ** 2, 4))
total_rewards = []

for episode in range(total_episodes):
    reward = 0
    i = 0
    j = 0
    state = (lake_size * i + j) #qtable[(lake_size * i) + j]
    #print (state)
    for step in range(max_steps):
#        print ('step num:', step)
        action_validity = True
        while (action_validity):
            action_taken = GetAction(state, epsilon, qtable)
            #print (action_taken)
            if action_taken == 0: # go left
                if j != 0:
                    j -= 1
                    action_validity = False
            elif action_taken == 1: # go right
                if j != lake_size - 1:
                    j += 1
                    action_validity = False
            elif action_taken == 2: # go up
                if i != 0:
                    i -= 1
                    action_validity = False
            elif action_taken == 3: #go down
                 if i != lake_size - 1:
                    i += 1
                    action_validity = False

        new_state = (lake_size * i) + j
        qtable[state, action_taken] = (1-learning_rate) * qtable[state, action_taken] + learning_rate * (env_lake[i][j] + gamma * numpy.max(qtable[new_state, :]))
        reward += env_lake[i][j]
        epsilon = min_epsilon + (max_epsilon-min_epsilon) * numpy.exp(-decay_rate*episode)
        if i == (lake_size-1) and j == (lake_size - 1): # reached goal state
            break
        else:
            state = new_state
    total_rewards.append(reward)

#print (total_rewards)
print (qtable)
plt.plot(total_rewards)
plt.show()
