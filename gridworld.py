"""Reinforcement learning agent that learns to navigate through a gridworld.
The grid is 4 * 3 in size. Agent starts from bottom left and can end in one of two positions - (3, 0) with reward +1
or (3, 1) with reward -1."""

import numpy as np

# global variables

board_rows = 3
board_columns = 4
win_state = (0, 3)
lose_state = (1, 3)
start = (2, 0)
deterministic = True

class gridWorld:
    def __init__(self, state = start):
        self.board = np.zeros([board_rows, board_columns])
        self.board[1, 1] = -1
        self.state = state
        self.isEnd = False
        self.deterministic = deterministic
        self.win_state = win_state
        self.lose_state = lose_state


    def giveReward(self):
        if self.state == self.win_state:
            return 1
        elif self.state == self.lose_state:
            return -1
        else:
            return 0

    def isEndFunc(self):
        if (self.state == win_state) or (self.state == lose_state):
            self.isEnd = True

    def nextPosition(self, action):
        """action: up, down, left, right
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position"""
        if self.deterministic:
            if action == "up":
                nxtstate = (self.state[0] - 1, self.state[1])
            elif action == "down":
                nxtstate = (self.state[0] + 1, self.state[1])
            elif action == "left":
                nxtstate = (self.state[0], self.state[1] - 1)
            else:
                nxtstate = (self.state[0], self.state[1] + 1)
            # if next state legal
            if (nxtstate[0] >= 0) and (nxtstate[0] <= 2):
                if (nxtstate[1] >= 0) and (nxtstate[1] <= 3):
                    if nxtstate != (1, 1):
                        return nxtstate

            return self.state


    def showBoard(self):
        self.board[self.state] = 1
        for i in range(0, board_rows):
            print('-----------------')
            out = '| '
            for j in range(0, board_columns):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')

class Agent:
    def __init__(self):
        self.states = []
        self.actions = ['up', 'down', 'left', 'right']
        self.state = gridWorld()
        self.lr = 0.2 # learning rate
        self.exp_rate = 0.3

        self.state_values = {}
        for i in range(board_rows):
                for j in range(board_columns):
                    self.state_values[(i, j)] = 0 # initialize all state values to 0

    def ChooseAction(self):
        mx_nxt_reward = 0
        action = ''

        if np.random.uniform(0, 1) <= self.exp_rate: # choose a random action
            action = np.random.choice(self.actions)

        else:
            # greedy action
            for a in self.actions:
                # if the action is deterministic
                next_reward = self.state_values[self.state.nextPosition(a)]
                if next_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = next_reward
        return action

    def play(self, rounds = 10):
        i = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.state.isEnd:
                # back propagate
                reward = self.state.giveReward()
                # explicitly assign end state to reward values
                self.state_values[self.state.state] = reward  # this is optional
                print("Game End Reward", reward)
                for s in reversed(self.states):
                    reward = self.state_values[s] + self.lr * (reward - self.state_values[s])
                    self.state_values[s] = round(reward, 3)
                self.reset()
                i += 1

            else:
                action = self.ChooseAction()
                # append trace
                self.states.append(self.state.nextPosition(action))
                print("current position {} action {}".format(self.state.state, action))
                # by taking the action, it reaches the next state
                self.state = self.takeAction(action)
                # mark is end
                self.state.isEndFunc()
                print("nxt state", self.state.state)
                print("---------------------")

    def showValues(self):
        for i in range(0, board_rows):
            print('----------------------------------')
            out = '| '
            for j in range(0, board_columns):
                out += str(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print('----------------------------------')

    def takeAction(self, action):
        position = self.state.nextPosition(action)
        return gridWorld(state=position)

    def reset(self):
        self.states = []
        self.state = gridWorld()



if __name__ == "__main__":
    ag = Agent()
    ag.play(50)
    print(ag.showValues())
