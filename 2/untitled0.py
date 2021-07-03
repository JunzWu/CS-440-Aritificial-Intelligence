# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 14:35:01 2019
False
@author: Junz
"""
from random import randint
import uttt2
ttt = uttt2.ultimateTicTacToe()
print(ttt.playGamePredifinedAgent(True,True,True)) 
#print(ttt.minimax(1, 0, False))
#print(ttt.alphabeta(1, 0, 0, 0, False))


'''
winner=0
bestMove=[]
bestValue=[]
gameBoards=[]
BoardIdx = 4
isMax = True
for i in range(8):
 #value1 = ttt.minimax(1, BoardIdx, isMax)
 value1 = ttt.alphabeta(1, BoardIdx, 0, 0, isMax)
 bestValue += [value1]
 i = ttt.value1.index(value1)
 ttt.value1 = []
 k = 0
 if isMax == True:
     for j in range(9):
         if ttt.board[int(BoardIdx/3)*3+int(j/3)][int(BoardIdx%3*3)+int(j%3)] == '_':
             if k == i:
                 ttt.board[int(BoardIdx/3)*3+int(j/3)][int(BoardIdx%3*3)+int(j%3)] = 'X'
                 i = j
                 break
             else:
                 k += 1   
 else:
    for j in range(9):
         if ttt.board[int(BoardIdx/3)*3+int(j/3)][int(BoardIdx%3*3)+int(j%3)] == '_':
             if k == i:
                 ttt.board[int(BoardIdx/3)*3+int(j/3)][int(BoardIdx%3*3)+int(j%3)] = 'O'
                 i = j
                 break
             else:
                 k += 1
 bestMove += [(int(BoardIdx/3)*3+int(i/3), int(BoardIdx%3*3)+int(i%3))]
 ttt.expandedNodes += 1
 BoardIdx = i
 isMax = not isMax


winner = ttt.checkWinner()
gameBoards = ttt.board
print(winner)
print(ttt.expandedNodes)
print(bestMove)
print(bestValue)
print(gameBoards)
print(BoardIdx)
'''