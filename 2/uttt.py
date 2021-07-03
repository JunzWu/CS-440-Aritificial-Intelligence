from time import sleep
from math import inf
from random import randint

class ultimateTicTacToe:
    def __init__(self):
        """
        Initialization of the game.
        """
        self.board=[['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_']]
        self.maxPlayer='X'
        self.minPlayer='O'
        self.maxDepth=3
        #The start indexes of each local board
        self.globalIdx=[(0,0),(0,3),(0,6),(3,0),(3,3),(3,6),(6,0),(6,3),(6,6)]

        #Start local board index for reflex agent playing
        self.startBoardIdx=4
        #self.startBoardIdx=randint(0,8)

        #utility value for reflex offensive and reflex defensive agents
        self.winnerMaxUtility=10000
        self.twoInARowMaxUtility=500
        self.preventThreeInARowMaxUtility=100
        self.cornerMaxUtility=30

        self.winnerMinUtility=-10000
        self.twoInARowMinUtility=-100
        self.preventThreeInARowMinUtility=-500
        self.cornerMinUtility=-30

        self.expandedNodes=0
        self.currPlayer=True
        
        self.value1 = []
        self.value2 = []
        self.value3 = []

    def printGameBoard(self):
        """
        This function prints the current game board.
        """
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[:3]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[3:6]])+'\n')
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.board[6:9]])+'\n')


    def evaluateDesigned(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for predifined agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        #YOUR CODE HERE
        score=0
        if isMax == True:
            for i in range(9):
                # first rule
                # situation of diagnol line
                win = True
                for k in range(3):
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+k)] != 'X' :
                            win = False
                if win == True:
                    score = 10000
                    
                win = True    
                for k in range(3):
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+2-k)] != 'X' :
                            win = False
                if win == True:
                    score = 10000

                for j in range(3):
                    # situation of horizontal line
                    win = True
                    for k in range(3):
                        if self.board[int(int(i/3)*3+j)][int(i%3*3+k)] != 'X' :
                            win = False
                    if win == True:
                        score = 10000
                        
                    # situation of vertical line
                    win = True
                    for k in range(3):
                        if self.board[int(int(i/3)*3+k)][int(i%3*3+j)] != 'X' :
                            win = False
                    if win == True:
                        score = 10000
            if score ==0:
              for i in range(9):    
                # second rule
                nX = 0
                nO = 0
                for k in range(3):
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+k)] == 'X' :
                            nX += 1
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+k)] == 'O' :
                            nO += 1
                if nX == 2 and nO == 0:
                    score += 500
                if nX == 1 and nO == 2:
                    score += 100
                
                nX = 0
                nO = 0
                for k in range(3):
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+2-k)] == 'X' :
                            nX += 1
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+2-k)] == 'O' :
                            nO += 1
                if nX == 2 and nO == 0:
                    score += 500
                if nX == 1 and nO == 2:
                    score += 100
                
                for j in range(3):
                    nX = 0
                    nO = 0
                    for k in range(3):
                        if self.board[int(int(i/3)*3+j)][int(i%3*3+k)] == 'X' :
                            nX += 1
                        if self.board[int(int(i/3)*3+j)][int(i%3*3+k)] == 'O' :
                            nO += 1
                    if nX == 2 and nO == 0:
                        score += 500
                    if nX == 1 and nO == 2:
                        score += 100
                    
                    nX = 0
                    nO = 0
                    for k in range(3):
                        if self.board[int(int(i/3)*3+k)][int(i%3*3+j)] == 'X' :
                            nX += 1
                        if self.board[int(int(i/3)*3+k)][int(i%3*3+j)] == 'O' :
                            nO += 1
                    if nX == 2 and nO == 0:
                        score += 500
                    if nX == 1 and nO == 2:
                        score += 100
            # third rule
            if score == 0: 
                for i in range(9):
                    if self.board[int(int(i/3)*3)][int(i%3*3)] == 'X' :
                        score += 30
                    if self.board[int(int(i/3)*3)+2][int(i%3*3)] == 'X' :
                        score += 30
                    if self.board[int(int(i/3)*3)][int(i%3*3)+2] == 'X' :
                        score += 30
                    if self.board[int(int(i/3)*3)+2][int(i%3*3)+2] == 'X' :
                        score += 30
        else:
            for i in range(9):
                # first rule
                # situation of diagnol line
                win = True
                lose=True
                for k in range(3):
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+k)] != 'O' :
                            win = False
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+k)] != 'X' :
                            lose = False
                if win == True:
                    score = -10000
                if lose==True:
                    score= 10000
                
                win = True
                lose=True
                for k in range(3):
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+2-k)] != 'O' :
                            win = False
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+2-k)] != 'X' :
                            lose = False
                if win == True:
                    score = -10000
                if lose==True:
                    score= 10000
                    
                for j in range(3):
                    # situation of horizontal line
                    win = True
                    for k in range(3):
                        if self.board[int(int(i/3)*3+j)][int(i%3*3+k)] != 'O' :
                            win = False
                        if self.board[int(int(i/3)*3+j)][int(i%3*3+k)] != 'X' :
                            lose = False
                    if win == True:
                        score = -10000
                    if lose==True:
                        score= 10000
                    # situation of vertical line
                    win = True
                    for k in range(3):
                        if self.board[int(int(i/3)*3+k)][int(i%3*3+j)] != 'O' :
                            win = False
                        if self.board[int(int(i/3)*3+k)][int(i%3*3+j)] != 'X' :
                            lose = False
                    if win == True:
                        score = -10000
                    if lose==True:
                        score= 10000
            if score ==0:
              for i in range(9):    
                # second rule
                nX = 0
                nO = 0
                for k in range(3):
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+k)] == 'X' :
                            nX += 1
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+k)] == 'O' :
                            nO += 1
                if nX == 0 and nO == 2:
                    score -= 500
                if nX == 2 and nO == 1:
                    score -= 150
                if nX == 2 and nO == 0:
                    score += 460
                    
                nX = 0
                nO = 0
                for k in range(3):
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+2-k)] == 'X' :
                            nX += 1
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+2-k)] == 'O' :
                            nO += 1
                if nX == 0 and nO == 2:
                    score -= 500
                if nX == 2 and nO == 1:
                    score -= 150
                if nX == 2 and nO == 0:
                    score += 460
                
                for j in range(3):
                    nX = 0
                    nO = 0
                    for k in range(3):
                        if self.board[int(int(i/3)*3+j)][int(i%3*3+k)] == 'X' :
                            nX += 1
                        if self.board[int(int(i/3)*3+j)][int(i%3*3+k)] == 'O' :
                            nO += 1
                    if nX == 0 and nO == 2:
                        score -= 500
                    if nX == 2 and nO == 1:
                        score -= 150
                    if nX == 2 and nO == 0:
                        score += 460
                    
                    nX = 0
                    nO = 0
                    for k in range(3):
                        if self.board[int(int(i/3)*3+k)][int(i%3*3+j)] == 'X' :
                            nX += 1
                        if self.board[int(int(i/3)*3+k)][int(i%3*3+j)] == 'O' :
                            nO += 1
                    if nX == 0 and nO == 2:
                        score -= 500
                    if nX == 2 and nO == 1:
                        score -= 150
                    if nX == 2 and nO == 0:
                        score += 460
            if score == 0:            
                for i in range(9):
                #third rule    
                    if self.board[int(int(i/3)*3)][int(i%3*3)] == 'O' :
                        score -= 30
                    if self.board[int(int(i/3)*3)+2][int(i%3*3)] == 'O' :
                        score -= 30
                    if self.board[int(int(i/3)*3)][int(i%3*3)+2] == 'O' :
                        score -= 30
                    if self.board[int(int(i/3)*3)+2][int(i%3*3)+2] == 'O' :
                        score -= 30
        return score


    def evaluatePredifined(self, isMax):
        """
        This function implements the evaluation function for ultimate tic tac toe for your own agent.
        input args:
        isMax(bool): boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        score(float): estimated utility score for maxPlayer or minPlayer
        """
        #YOUR CODE HERE
        score=0
        if isMax == True:
            for i in range(9):
                # first rule
                # situation of diagnol line
                win = True
                for k in range(3):
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+k)] != 'X' :
                            win = False
                if win == True:
                    score += 10000
                    
                win = True    
                for k in range(3):
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+2-k)] != 'X' :
                            win = False
                if win == True:
                    score += 10000

                for j in range(3):
                    # situation of horizontal line
                    win = True
                    for k in range(3):
                        if self.board[int(int(i/3)*3+j)][int(i%3*3+k)] != 'X' :
                            win = False
                    if win == True:
                        score += 10000
                        
                    # situation of vertical line
                    win = True
                    for k in range(3):
                        if self.board[int(int(i/3)*3+k)][int(i%3*3+j)] != 'X' :
                            win = False
                    if win == True:
                        score += 10000
            if score < 10000:
              for i in range(9):    
                # second rule
                nX = 0
                nO = 0
                for k in range(3):
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+k)] == 'X' :
                            nX += 1
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+k)] == 'O' :
                            nO += 1
                if nX == 2 and nO == 0:
                    score += 500
                if nX == 1 and nO == 2:
                    score += 100
                
                nX = 0
                nO = 0
                for k in range(3):
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+2-k)] == 'X' :
                            nX += 1
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+2-k)] == 'O' :
                            nO += 1
                if nX == 2 and nO == 0:
                    score += 500
                if nX == 1 and nO == 2:
                    score += 100
                
                for j in range(3):
                    nX = 0
                    nO = 0
                    for k in range(3):
                        if self.board[int(int(i/3)*3+j)][int(i%3*3+k)] == 'X' :
                            nX += 1
                        if self.board[int(int(i/3)*3+j)][int(i%3*3+k)] == 'O' :
                            nO += 1
                    if nX == 2 and nO == 0:
                        score += 500
                    if nX == 1 and nO == 2:
                        score += 100
                    
                    nX = 0
                    nO = 0
                    for k in range(3):
                        if self.board[int(int(i/3)*3+k)][int(i%3*3+j)] == 'X' :
                            nX += 1
                        if self.board[int(int(i/3)*3+k)][int(i%3*3+j)] == 'O' :
                            nO += 1
                    if nX == 2 and nO == 0:
                        score += 500
                    if nX == 1 and nO == 2:
                        score += 100
            
            if score == 0: 
                for i in range(9):
                    if self.board[int(int(i/3)*3)][int(i%3*3)] == 'X' :
                        score += 30
                    if self.board[int(int(i/3)*3)+2][int(i%3*3)] == 'X' :
                        score += 30
                    if self.board[int(int(i/3)*3)][int(i%3*3)+2] == 'X' :
                        score += 30
                    if self.board[int(int(i/3)*3)+2][int(i%3*3)+2] == 'X' :
                        score += 30
                
        else:
            for i in range(9):
                # first rule
                # situation of diagnol line
                win = True
                for k in range(3):
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+k)] != 'O' :
                            win = False
                if win == True:
                    score -= 10000
                
                win = True
                for k in range(3):
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+2-k)] != 'O' :
                            win = False
                if win == True:
                    score -= 10000

                for j in range(3):
                    # situation of horizontal line
                    win = True
                    for k in range(3):
                        if self.board[int(int(i/3)*3+j)][int(i%3*3+k)] != 'O' :
                            win = False
                    if win == True:
                        score -= 10000
                    # situation of vertical line
                    win = True
                    for k in range(3):
                        if self.board[int(int(i/3)*3+k)][int(i%3*3+j)] != 'O' :
                            win = False
                    if win == True:
                        score -= 10000
            if score > -10000:
              for i in range(9):    
                # second rule
                nX = 0
                nO = 0
                for k in range(3):
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+k)] == 'X' :
                            nX += 1
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+k)] == 'O' :
                            nO += 1
                if nX == 0 and nO == 2:
                    score -= 100
                if nX == 2 and nO == 1:
                    score -= 500
                    
                nX = 0
                nO = 0
                for k in range(3):
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+2-k)] == 'X' :
                            nX += 1
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+2-k)] == 'O' :
                            nO += 1
                if nX == 0 and nO == 2:
                    score -= 100
                if nX == 2 and nO == 1:
                    score -= 500
                
                for j in range(3):
                    nX = 0
                    nO = 0
                    for k in range(3):
                        if self.board[int(int(i/3)*3+j)][int(i%3*3+k)] == 'X' :
                            nX += 1
                        if self.board[int(int(i/3)*3+j)][int(i%3*3+k)] == 'O' :
                            nO += 1
                    if nX == 0 and nO == 2:
                        score -= 100
                    if nX == 2 and nO == 1:
                        score -= 500
                    
                    nX = 0
                    nO = 0
                    for k in range(3):
                        if self.board[int(int(i/3)*3+k)][int(i%3*3+j)] == 'X' :
                            nX += 1
                        if self.board[int(int(i/3)*3+k)][int(i%3*3+j)] == 'O' :
                            nO += 1
                    if nX == 0 and nO == 2:
                        score -= 100
                    if nX == 2 and nO == 1:
                        score -= 500
            if score == 0:            
                for i in range(9):
                #third rule    
                    if self.board[int(int(i/3)*3)][int(i%3*3)] == 'O' :
                        score -= 30
                    if self.board[int(int(i/3)*3)+2][int(i%3*3)] == 'O' :
                        score -= 30
                    if self.board[int(int(i/3)*3)][int(i%3*3)+2] == 'O' :
                        score -= 30
                    if self.board[int(int(i/3)*3)+2][int(i%3*3)+2] == 'O' :
                        score -= 30
        
        return score

    def checkMovesLeft(self):
        """
        This function checks whether any legal move remains on the board.
        output:
        movesLeft(bool): boolean variable indicates whether any legal move remains
                        on the board.
        """
        #YOUR CODE HERE
        movesLeft=False
        for i in range(9):
            for j in range(3):
                for k in range(3):
                    if self.board[int(int(i/3)*3+j)][int(i%3*3+k)] == '_' :
                            movesLeft=True
        return movesLeft

    def checkWinner(self):
        #Return termimnal node status for maximizer player 1-win,0-tie,-1-lose
        """
        This function checks whether there is a winner on the board.
        output:
        winner(int): Return 0 if there is no winner.
                     Return 1 if maxPlayer is the winner.
                     Return -1 if miniPlayer is the winner.
        """
        #YOUR CODE HERE
        winner = 0
        for i in range(9):
                # first rule
                # situation of diagnol line
                nX = 0
                nO = 0
                for k in range(3):
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+k)] == 'X' :
                            nX += 1
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+k)] == 'O' :
                            nO += 1
                if nX == 3:
                    winner = 1
                    break
                if nO == 3:
                    winner = -1
                    break
                
                nX = 0
                nO = 0
                for k in range(3):
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+2-k)] == 'X' :
                            nX += 1
                    if self.board[int(int(i/3)*3+k)][int(i%3*3+2-k)] == 'O' :
                            nO += 1
                if nX == 3:
                    winner = 1
                    break
                if nO == 3:
                    winner = -1
                    break
                
                for j in range(3):
                    nX = 0
                    nO = 0
                    for k in range(3):
                        if self.board[int(int(i/3)*3+j)][int(i%3*3+k)] == 'X' :
                            nX += 1
                        if self.board[int(int(i/3)*3+j)][int(i%3*3+k)] == 'O' :
                            nO += 1
                    if nX == 3:
                        winner = 1
                        break
                    if nO == 3:
                        winner = -1
                        break
                    
                    nX = 0
                    nO = 0
                    for k in range(3):
                        if self.board[int(int(i/3)*3+k)][int(i%3*3+j)] == 'X' :
                            nX += 1
                        if self.board[int(int(i/3)*3+k)][int(i%3*3+j)] == 'O' :
                            nO += 1
                    if nX == 3:
                        winner = 1
                        break
                    if nO == 3:
                        winner = -1
                        break
                if winner != 0:
                    break

        
        return winner

    def designedalphabeta(self,depth,currBoardIdx,alpha,beta,isMax):
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        #YOUR CODE HERE
        i = 0
        if depth == 4:
            return self.evaluateDesigned(isMax)
        for j in range(3):
            for k in range(3):
                if self.board[int(currBoardIdx/3)*3+j][int(currBoardIdx%3*3)+k] == '_':
                    if isMax == True:
                        self.board[int(currBoardIdx/3)*3+j][int(currBoardIdx%3*3)+k] = 'X'
                    else:
                        self.board[int(currBoardIdx/3)*3+j][int(currBoardIdx%3*3)+k] = 'O'
                    if depth == 3:
                        self.expandedNodes += 1
                        self.value3 += [self.designedalphabeta(depth+1, 3*j+k, alpha, beta, isMax)]
                    if depth == 2:
                      self.expandedNodes += 1
                      if isMax == False:
                        if alpha != 0:
                            if self.designedalphabeta(depth+1, 3*j+k, alpha, beta, not isMax) == 1:
                                i = 1
                                self.board[int(currBoardIdx/3)*3+j][int(currBoardIdx%3*3)+k] = '_'
                                break
                        if alpha == 0:
                            self.designedalphabeta(depth+1, 3*j+k, alpha, beta, not isMax)
                      if isMax == True:
                        if self.checkWinner()==1:
                            self.value2+=[10000]
                            self.board[int(currBoardIdx/3)*3+j][int(currBoardIdx%3*3)+k] = '_'
                            continue
                        if beta != 0:
                            if self.designedalphabeta(depth+1, 3*j+k, alpha, beta, not isMax) == 1:
                                i = 1
                                self.board[int(currBoardIdx/3)*3+j][int(currBoardIdx%3*3)+k] = '_'
                                break
                        if beta == 0:
                            self.designedalphabeta(depth+1, 3*j+k, alpha, beta, not isMax)
                    if depth == 1:
                        if isMax == True:
                            self.expandedNodes += 1
                            alpha = self.designedalphabeta(depth+1, 3*j+k, alpha, beta, not isMax)
                        else:
                            self.expandedNodes += 1
                            if self.checkWinner()==-1:
                                self.value1+=[-10000]
                                self.board[int(currBoardIdx/3)*3+j][int(currBoardIdx%3*3)+k] = '_'
                                beta=-10000
                                continue
                            beta = self.designedalphabeta(depth+1, 3*j+k, alpha, beta, not isMax)
                        
                    self.board[int(currBoardIdx/3)*3+j][int(currBoardIdx%3*3)+k] = '_'
            if i == 1:
                break
            
                    
        if depth == 3:
            if isMax == True:
                self.value2 += [max(self.value3)]
                if max(self.value3) <= alpha:
                    self.value3 = []
                    return 1
                else:
                    self.value3 = []
                    return 0
            else:
                self.value2 += [min(self.value3)]
                if min(self.value3) >= beta:
                    self.value3 = []
                    return 1
                else:
                    self.value3 = []
                    return 0
        if depth == 2:
            if isMax == False:
                self.value1 += [min(self.value2)]
                #print(alpha)
                if alpha < min(self.value2):
                    alpha = min(self.value2)
                #print(self.value2)
                #print(alpha)
                self.value2 = []
                return alpha
            else:
                self.value1 += [max(self.value2)]
                #print(beta)
                if beta > max(self.value2):
                    beta = max(self.value2)
                #print(self.value2)
                #print(beta)
                self.value2 = []
                return beta
        if depth == 1:
            if isMax == True:
                bestValue = max(self.value1)
            else:
                bestValue = min(self.value1)
            return bestValue


    def alphabeta(self,depth,currBoardIdx,alpha,beta,isMax):
        """
        This function implements alpha-beta algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        alpha(float): alpha value
        beta(float): beta value
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        #YOUR CODE HERE
        i = 0
        if depth == 4:            
            return self.evaluatePredifined(isMax)
        for j in range(3):
            for k in range(3):
                if self.board[int(currBoardIdx/3)*3+j][int(currBoardIdx%3*3)+k] == '_':
                    if isMax == True:
                        self.board[int(currBoardIdx/3)*3+j][int(currBoardIdx%3*3)+k] = 'X'
                    else:
                        self.board[int(currBoardIdx/3)*3+j][int(currBoardIdx%3*3)+k] = 'O'
                    if depth == 3:
                        self.expandedNodes += 1
                        self.value3 += [self.alphabeta(depth+1, 3*j+k, alpha, beta, isMax)]
                    if depth == 2:
                      self.expandedNodes += 1
                      if isMax == False:
                        if alpha != 0:
                            if self.alphabeta(depth+1, 3*j+k, alpha, beta, not isMax) == 1:
                                i = 1
                                self.board[int(currBoardIdx/3)*3+j][int(currBoardIdx%3*3)+k] = '_'
                                break
                        if alpha == 0:
                            self.alphabeta(depth+1, 3*j+k, alpha, beta, not isMax)
                      if isMax == True:      
                        if beta != 0:
                            if self.alphabeta(depth+1, 3*j+k, alpha, beta, not isMax) == 1:
                                i = 1
                                self.board[int(currBoardIdx/3)*3+j][int(currBoardIdx%3*3)+k] = '_'
                                break
                        if beta == 0:
                            self.alphabeta(depth+1, 3*j+k, alpha, beta, not isMax)
                    if depth == 1:
                        if isMax == True:
                            self.expandedNodes += 1
                            alpha = self.alphabeta(depth+1, 3*j+k, alpha, beta, not isMax)
                        else:
                            self.expandedNodes += 1
                            beta = self.alphabeta(depth+1, 3*j+k, alpha, beta, not isMax)
                        
                    self.board[int(currBoardIdx/3)*3+j][int(currBoardIdx%3*3)+k] = '_'
            if i == 1:
                break
            
                    
        if depth == 3:
            if isMax == True:
                self.value2 += [max(self.value3)]
                if max(self.value3) <= alpha:
                    self.value3 = []
                    return 1
                else:
                    self.value3 = []
                    return 0
            else:
                self.value2 += [min(self.value3)]
                if min(self.value3) >= beta:
                    self.value3 = []
                    return 1
                else:
                    self.value3 = []
                    return 0
        if depth == 2:
            if isMax == False:
                self.value1 += [min(self.value2)]
                #print(alpha)
                if alpha < min(self.value2):
                    alpha = min(self.value2)
                #print(self.value2)
                #print(alpha)
                self.value2 = []
                return alpha
            else:
                self.value1 += [max(self.value2)]
                #print(beta)
                if beta > max(self.value2):
                    beta = max(self.value2)
                #print(self.value2)
                #print(beta)
                self.value2 = []
                return beta
        if depth == 1:
            if isMax == True:
                bestValue = max(self.value1)
            else:
                bestValue = min(self.value1)
            return bestValue

    def minimax(self, depth, currBoardIdx, isMax):
        """
        This function implements minimax algorithm for ultimate tic-tac-toe game.
        input args:
        depth(int): current depth level
        currBoardIdx(int): current local board index
        isMax(bool):boolean variable indicates whether it's maxPlayer or minPlayer.
                     True for maxPlayer, False for minPlayer
        output:
        bestValue(float):the bestValue that current player may have
        """
        #YOUR CODE HERE
        if depth == 4:
            self.expandedNodes += 1
            return self.evaluatePredifined(isMax)
        for j in range(3):
            for k in range(3):
                if self.board[int(currBoardIdx/3)*3+j][int(currBoardIdx%3*3)+k] == '_':
                    if isMax == True:
                        self.board[int(currBoardIdx/3)*3+j][int(currBoardIdx%3*3)+k] = 'X'
                    else:
                        self.board[int(currBoardIdx/3)*3+j][int(currBoardIdx%3*3)+k] = 'O'
                    if depth == 3:
                        self.value3 += [self.minimax(depth+1, 3*j+k, isMax)]
                    if depth == 2:
                        self.expandedNodes += 1
                        self.minimax(depth+1, 3*j+k, not isMax)
                    if depth == 1:
                        self.expandedNodes += 1
                        self.minimax(depth+1, 3*j+k, not isMax)
                    self.board[int(currBoardIdx/3)*3+j][int(currBoardIdx%3*3)+k] = '_'
                    
        if depth == 3:
            if isMax == True:
                self.value2 += [max(self.value3)]
            else:
                self.value2 += [min(self.value3)]
            self.value3 = []
            return
        if depth == 2:
            if isMax == False:
                self.value1 += [min(self.value2)]
                #print(self.value2)
            else:
                self.value1 += [max(self.value2)]
                #print(self.value2)
            self.value2 = []
            return
        if depth == 1:
            if isMax == True:
                bestValue = max(self.value1)
            else:
                bestValue = min(self.value1)
            return bestValue

    def playGamePredifinedAgent(self,maxFirst,isMinimaxOffensive,isMinimaxDefensive):
        """
        This function implements the processes of the game of predifined offensive agent vs defensive agent.
        input args:
        maxFirst(bool): boolean variable indicates whether maxPlayer or minPlayer plays first.
                        True for maxPlayer plays first, and False for minPlayer plays first.
        isMinimaxOffensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for offensive agent.
                        True is minimax and False is alpha-beta.
        isMinimaxDefensive(bool):boolean variable indicates whether it's using minimax or alpha-beta pruning algorithm for defensive agent.
                        True is minimax and False is alpha-beta.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        bestValue(list of float): list of bestValue at each move
        expandedNodes(list of int): list of expanded nodes at each move
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        winner=0
        bestMove=[]
        bestValue=[]
        gameBoards=[]
        expandedNodes = []
        BoardIdx = 4

        if maxFirst == True:
            isMax = True
        else:
            isMax = False
        while self.checkWinner() == 0:
            if self.checkMovesLeft() == False:
                break
            if isMinimaxOffensive == True:
                if isMinimaxDefensive == True:
                    value1 = self.minimax(1, BoardIdx, isMax)
                else:
                    if isMax == True:
                        value1 = self.minimax(1, BoardIdx, isMax)
                    else:
                        value1 = self.alphabeta(1, BoardIdx, 0, 0, isMax)
            else:
                if isMinimaxDefensive == True:
                    if isMax == True:
                        value1 = self.alphabeta(1, BoardIdx, 0, 0, isMax)
                    else:
                        value1 = self.minimax(1, BoardIdx, isMax)
                else:
                    value1 = self.alphabeta(1, BoardIdx, 0, 0, isMax)
            
            expandedNodes += [self.expandedNodes]
            self.expandedNodes = 0
            bestValue += [value1]
            i = self.value1.index(value1)
            self.value1 = []
            k = 0
            if isMax == True:
                for j in range(9):
                    if self.board[int(BoardIdx/3)*3+int(j/3)][int(BoardIdx%3*3)+int(j%3)] == '_':
                        if k == i:
                            self.board[int(BoardIdx/3)*3+int(j/3)][int(BoardIdx%3*3)+int(j%3)] = 'X'
                            i = j
                            break
                        else:
                            k += 1 
            else:
                 for j in range(9):
                     if self.board[int(BoardIdx/3)*3+int(j/3)][int(BoardIdx%3*3)+int(j%3)] == '_':
                         if k == i:
                             self.board[int(BoardIdx/3)*3+int(j/3)][int(BoardIdx%3*3)+int(j%3)] = 'O'
                             i = j
                             break
                         else:
                             k += 1 
            bestMove += [(int(BoardIdx/3)*3+int(i/3), int(BoardIdx%3*3)+int(i%3))]
            BoardIdx = i
            isMax = not isMax
        winner = self.checkWinner()
        gameBoards = self.board
        self.board=[['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_']]

        

        return gameBoards, bestMove, expandedNodes, bestValue, winner
        
    def playGameYourAgent(self):
        """
        This function implements the processes of the game of your own agent vs predifined offensive agent.
        input args:
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove=[]
        gameBoards=[]
        expandedNodes = []
        bestValue=[]
        winner=0
        
        maxFirst = randint(0,1)
        BoardIdx = randint(0,8)
        if maxFirst == 1:
            isMax = True
        if maxFirst == 0:
            isMax = False
        if isMax:
            print("Maxplayer goes first!")
        else:
            print("Minplayer goes first!")
        
        print("startidx=%d"%BoardIdx)
        while self.checkWinner() == 0:
            if self.checkMovesLeft() == False:
                break
            value1 = self.designedalphabeta(1, BoardIdx, 0, 0, isMax)
            expandedNodes += [self.expandedNodes]
            self.expandedNodes = 0
            bestValue += [value1]
            i = self.value1.index(value1)
            self.value1 = []
            k = 0
            if isMax == True:
                for j in range(9):
                    if self.board[int(BoardIdx/3)*3+int(j/3)][int(BoardIdx%3*3)+int(j%3)] == '_':
                        if k == i:
                            self.board[int(BoardIdx/3)*3+int(j/3)][int(BoardIdx%3*3)+int(j%3)] = 'X'
                            i = j
                            break
                        else:
                            k += 1 
            else:
                 for j in range(9):
                     if self.board[int(BoardIdx/3)*3+int(j/3)][int(BoardIdx%3*3)+int(j%3)] == '_':
                         if k == i:
                             self.board[int(BoardIdx/3)*3+int(j/3)][int(BoardIdx%3*3)+int(j%3)] = 'O'
                             i = j
                             break
                         else:
                             k += 1 
            bestMove += [(int(BoardIdx/3)*3+int(i/3), int(BoardIdx%3*3)+int(i%3))]
            BoardIdx = i
            isMax = not isMax
        winner = self.checkWinner()
        gameBoards = self.board
        self.board=[['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_']]
        return gameBoards,bestMove, winner


    def playGameHuman(self):
        """
        This function implements the processes of the game of your own agent vs a human.
        output:
        bestMove(list of tuple): list of bestMove coordinates at each step
        gameBoards(list of 2d lists): list of game board positions at each move
        winner(int): 1 for maxPlayer is the winner, -1 for minPlayer is the winner, and 0 for tie.
        """
        #YOUR CODE HERE
        bestMove=[]
        gameBoards=[]
        winner=0
        
        maxFirst = randint(0,1)
        BoardIdx = randint(0,8)
        print("Start BoardIdx is %d"%BoardIdx)
        if maxFirst == 1:
            isMax = True
            print("you go first!")
        if maxFirst == 0:
            isMax = False
            print("agent goes first!")
        while self.checkWinner() == 0:
           if self.checkMovesLeft() == False:
            break
           if isMax == False:
                value1 = self.designedalphabeta(1, BoardIdx,0,0, isMax)
                #value1 = self.alphabeta(1, BoardIdx, 0, 0, isMax)
                i = self.value1.index(value1)
                self.value1 = []
                k = 0
                for j in range(9):
                    if self.board[int(BoardIdx/3)*3+int(j/3)][int(BoardIdx%3*3)+int(j%3)] == '_':
                        if k == i:
                            self.board[int(BoardIdx/3)*3+int(j/3)][int(BoardIdx%3*3)+int(j%3)] = 'O'
                            i = j
                            break
                        else:
                            k += 1 
                bestMove += [(int(BoardIdx/3)*3+int(i/3), int(BoardIdx%3*3)+int(i%3))]
                BoardIdx = i
           else:
               i,j = eval(input("please input the coordinate where you want to put X,Y:"))
               while self.board[i][j] != '_' or int(i/3)*3+int(j/3)!=BoardIdx:
                   print("this coordinate already has point or not valid")
                   i,j = eval(input("please input another coordinate X,Y:"))
               self.board[i][j] = 'X'
               BoardIdx=(i%3)*3+j%3
               bestMove+=[(i,j)]
           self.printGameBoard()
           isMax=not isMax
               
        
        winner = self.checkWinner()
        gameBoards = self.board
        self.board=[['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_'],
                    ['_','_','_','_','_','_','_','_','_']]
        return gameBoards, bestMove, winner

if __name__=="__main__":
    uttt=ultimateTicTacToe()
    gameBoards, bestMove, winner=uttt.playGameHuman()
    print(bestMove)
   # print(expandedNodes)
    print('\n'.join([' '.join([str(cell) for cell in row]) for row in gameBoards[:3]])+'\n')
    print('\n'.join([' '.join([str(cell) for cell in row]) for row in gameBoards[3:6]])+'\n')
    print('\n'.join([' '.join([str(cell) for cell in row]) for row in gameBoards[6:9]])+'\n')
    if winner == 1:
        print("The winner is maxPlayer!!!")
    elif winner == -1:
        print("The winner is minPlayer!!!")
    else:
        print("Tie. No winner:(")
