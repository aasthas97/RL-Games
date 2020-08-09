import numpy as np
import pickle

class TicTacToe:
    def __init__(self, playerOne, playerTwo, grid_size):
        """grid_size: size of tic tac toe grid. Default value is 3 (forms 3 x 3 grid)"""

        self.playerOne = playerOne
        self.playerTwo = playerTwo
        self.grid_size = grid_size
        self.board = np.empty([grid_size, grid_size], dtype = object)
        self.isEnd = False
        # playerOne plays first
        self.playerSymbol = 'x'
        self.boardState = None

    def reset(self):
        self.board = np.empty([grid_size, grid_size], dtype = object)
        self.boardState = None
        self.isEnd = False
        self.playerSymbol = 'x'

    def getState(self):
        """Get current state of the board"""
        return str(self.board.reshape(grid_size * grid_size))

    def vacantPositions(self):
        positions = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.board[i, j] == None:
                    positions.append((i, j))

        return positions

    def makeMove(self, position):
        self.board[position] = self.playerSymbol
        # change player
        self.playerSymbol = 'o' if self.playerSymbol == 'x' else 'x'

    def win(self):
        #check row
        for i in range(self.grid_size):
            if np.all(self.board[i,:]== 'x') == True:
                self.isend = True
                return 'x'
            if np.all(self.board[i,:]== 'o') == True:
                self.isend = True
                return 'o'

        # check column
        for j in range(self.grid_size):
            if np.all(self.board[:,j]== 'x') == True:
                self.isend = True
                return 'x'
            if np.all(self.board[:,j]== 'o') == True:
                self.isend = True
                return 'o'

        # check diagonal
        diagonal_x1 = np.all([np.all(self.board[i,i] == 'x') for i in range(self.grid_size)])
        diagonal_o1 = np.all([np.all(self.board[i,i] == 'o') for i in range(self.grid_size)])
        diagonal_x2 = np.all([np.all(self.board[i, self.grid_size - i - 1] == 'x') for i in range(self.grid_size)])
        diagonal_o2 = np.all([np.all(self.board[i, self.grid_size - i - 1] == 'o') for i in range(self.grid_size)])

        if diagonal_x1 == True or diagonal_x2 == True:
            self.isend = True
            return 'x'
        elif diagonal_o1 == True or diagonal_o2 == True:
            self.isend = True
            return 'o'

        # tie
        if len(self.vacantPositions()) == 0:
            self.isEnd = True
            return 'tie'


        # not end
        self.isEnd = False
        return None

    def rewards(self):
        result = self.win()
        if result == 'x':
            self.playerOne.giveReward(1)
            self.playerTwo.giveReward(-1)
        elif result == 'o':
            self.playerOne.giveReward(-1)
            self.playerTwo.giveReward(1)
        else: # tie or not end
            self.playerOne.giveReward(0.5)
            self.playerTwo.giveReward(0.5)

    def play(self, rounds = 100):
        for roundnumber in range(rounds):
            while not self.isEnd:
                # Player 1
                positions = self.vacantPositions()
                p1_action = self.playerOne.act(positions, self.board, self.playerSymbol)
                # take action and update board state
                self.makeMove(p1_action)
                board_state = self.getState()
                self.playerOne.addState(board_state)


                win = self.win()
                if win is not None: # ended with p1 either win or draw
                    self.rewards()
                    self.playerOne.reset()
                    self.playerTwo.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.vacantPositions()
                    p2_action = self.playerTwo.act(positions, self.board, self.playerSymbol)
                    self.makeMove(p2_action)
                    board_state = self.getState()
                    self.playerTwo.addState(board_state)

                    win = self.win()
                    if win is not None:
                        self.rewards()
                        self.playerOne.reset()
                        self.playerTwo.reset()
                        self.reset()
                        break


class Player:
    def __init__(self, name, exp_rate = 0.2):
        self.name = name
        self.states = []  # record all positions taken
        self.alpha = 0.4
        self.exp_rate = exp_rate # exploration rate, agent will explore 20% of the time
        self.decay_gamma = 0.8
        self.states_value = {}  # state -> value

    def getState(self, board):
        return str(board.reshape(grid_size * grid_size))

    def addState(self, state):
        self.states.append(state)

    def act(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate: # random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else: # greedy action
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardState = self.getState(next_board)
                if self.states_value.get(next_boardState) == None:
                    value = 0
                else:
                    value = self.states_value.get(next_boardState)

                if value >= value_max:
                    value_max = value
                    action = p

#         print("{} takes action {}".format(self.name, action))
        return action

    def giveReward(self, reward):
        for thisstate in reversed(self.states):
            if self.states_value.get(thisstate) is None:
                self.states_value[thisstate] = 0
            self.states_value[thisstate] += self.alpha * (self.decay_gamma * reward - self.states_value[thisstate])
            reward = self.states_value[thisstate]

    def reset(self):
        self.states = []

    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()

grid_size = 3
# training
player1 = Player("p1")
player2 = Player("p2")
game = TicTacToe(player1, player2, grid_size)

print("training.\n...")
game.play()
