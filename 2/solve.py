# -*- coding: utf-8 -*-
import numpy as np
import queue


def solvealgorithmX(X, Y, solution=[]):
    if not X:
        yield list(solution)
    else:
        c = min(X, key=lambda c: len(X[c]))
        for r in list(X[c]):
            solution.append(r)
            cols = select(X, Y, r)
            for s in solvealgorithmX(X, Y, solution):
                yield s

            deselect(X, Y, r, cols)
            solution.pop()

def select(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols

def deselect(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)
    
"""
def update(variable, currboard, targetboard):
    tvalues=[0]*len(variable)
    for var in variable:
        values=[]
        #pair of available pent 0=orginal pent, 1=flip, 2=rotate, 3=flip&rotate (i, coord)
        for x in range(targetboard.shape[0]):    
            for y in range(targetboard.shape[1]):
                coord=(x,y)
                if check_valid(coord, var, currboard, targetboard)==True:
                    values.append((0,coord))
                tempv=np.flip(var)
                if check_valid(coord, tempv, currboard, targetboard)==True:
                    values.append((1,coord))
                tempv=np.rot90(var)
                if check_valid(coord, tempv, currboard, targetboard)==True:
                    values.append((2,coord))
                tempv=np.flip(tempv)
                if check_valid(coord, tempv, currboard, targetboard)==True:
                    values.append((3,coord))
        tvalues[get_pent_idx(var)]=values
    return tvalues


def check_valid(coord, var, currboard, targetboard):
    if (coord[0]+var.shape[0]+1)>targetboard.shape[0] or (coord[1]+var.shape[1]+1)>targetboard.shape[1]:
        return False
    for row in range(var.shape[0]):
        for col in range(var.shape[1]):
            if var[row][col]!=0:
                if targetboard[coord[0]+row][coord[1]+col]==0:   #0's 
                    return False
                if currboard[coord[0]+row][coord[1]+col]==1:  # overlap
                    return False
            
    return True

def Arc(variable, values, currboard):
    q=queue.Queue()
    for i in range(len(values)):
        for j in range(len(values)):
            q.put((i,j))
    
    while not q.empty():
        temp=q.get()
        if Remove_Inconsistent_Values(variable, temp, values, currboard):
            for i in range(len(values)):
                if i !=temp[0]:
                    q.put((i,temp[0]))


def Remove_Inconsistent_Values(variable, temp, values, currboard):
    removed=False
    domain=values[temp[0]]
    for x in domain:
        var=variable[temp[0]]
        coord=x[1]
        if x[0]==1:
            var=np.flip(var)
        if x[0]==2:
            var=np.rot90(var)
        if x[0]==3:
            var=np.flip(np.rot90(var))
        tempboard=np.array(currboard)
        for row in range(var.shape[0]):
            for col in range(var.shape[1]):
                if var[row][col]!=0:
                    tempboard[coord[0]+row][coord[1]+col]=1
        
        delete=True
        for y in values[temp[1]]:
            vary=variable[temp[1]]
            coordy=y[1]
            if y[0]==1:
                vary=np.flip(vary)
            if y[0]==2:
                vary=np.rot90(vary)
            if y[0]==3:
                vary=np.flip(np.rot90(vary))
            ava=1
            for row in range(vary.shape[0]):
                for col in range(vary.shape[1]):
                    if vary[row][col]!=0:
                        if tempboard[coordy[0]+row][coordy[1]+col]!=0:
                            ava=0
            if ava==1:
                delete=False
                break
        
        if delete:
            values[temp[0]].remove(x)
            removed=True
    return removed
        
        
    
def removeall(list_,element):
    listn=[]
    for v in list_:
        if not np.array_equal(v,element):
            listn.append(v)
    return listn
"""

def check(pent,board,coord):
    values=[]
    if (coord[0]+pent.shape[0])>board.shape[0] or (coord[1]+pent.shape[1])>board.shape[1]:
        return values
    
    for row in range(pent.shape[0]):
        for col in range(pent.shape[1]):
            if board[coord[0]+row][coord[1]+col]==0:
                if pent[row][col]!=0:
                    return []
            if board[coord[0]+row][coord[1]+col]==1:
                if pent[row][col]!=0:
                    values.append((coord[0]+row,coord[1]+col))
    return values
    

def buildXY(board,pents):
    X=[]
    len_pents=len(pents)
    for i in range(len_pents):
        X.append("p%d"%i)
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            if board[row][col]==1:
                X.append((row,col))
    
    Y={}
    for i in range(len_pents):
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                pent_list=[]
                for r in range(8): #0=original 1= rotate90, 2=r180,3 =r270, 4=flip, 5=r90+f, 6=r180+f, 7=r270+f
                    pent=pents[i]
                    if r==0:
                        pent=pents[i]
                    if r==1:
                        pent=np.rot90(pents[i])
                    if r==2:
                        pent=np.rot90(np.rot90(pents[i]))
                    if r==3:
                        pent=np.rot90(np.rot90(np.rot90(pents[i])))
                    if r==4:
                        pent=np.flip(pents[i],1)
                    if r==5:
                        pent=np.flip(np.rot90(pents[i]),1)
                    if r==6:
                        pent=np.flip(np.rot90(np.rot90(pents[i])),1)
                    if r==7:
                        pent=np.flip(np.rot90(np.rot90(np.rot90(pents[i]))),1)
                    pent_list.append(pent)
                
                for idx in range(len(pent_list)-1):
                    for idx_2 in range(idx+1,len(pent_list)):
                        if np.array_equal(pent_list[idx],pent_list[idx_2]):
                            pent_list[idx]=0
                
                for idx in range(len(pent_list)):
                    if type(pent_list[idx]) is np.ndarray:
                        pent_name="%d %d"%(i,idx)
                        pent_name=(pent_name,(row,col))
                        coord=(row,col)
                        values=check(pent_list[idx],board,coord)
                        if values==[]:
                            continue
                        values.insert(0,"p%d"%i)
                        Y[pent_name]=values
    
    X={j:set() for j in X}
    for i in Y:
        for j in Y[i]:
            X[j].add(i)
    
    return X,Y
                
        
def solve(board, pents):
    """
    This is the function you will implement. It will take in a numpy array of the board
    as well as a list of n tiles in the form of numpy arrays. The solution returned
    is of the form [(p1, (row1, col1))...(pn,  (rown, coln))]
    where pi is a tile (may be rotated or flipped), and (rowi, coli) is 
    the coordinate of the upper left corner of pi in the board (lowest row and column index 
    that the tile covers).
    
    -Use np.flip and np.rot90 to manipulate pentominos.
    
    -You can assume there will always be a solution.
    """
    X,Y=buildXY(board,pents)

    solution=[]
    solution=solvealgorithmX(X,Y,solution)
    solution=next(solution)

    output=[]
    for var in solution:
        s=var[0]
        split=str.split(s,' ')
        i=int(split[0])
        r=int(split[1])
        pent=pents[i]
        if r==0:
            pent=pents[i]
        if r==1:
            pent=np.rot90(pents[i])
        if r==2:
            pent=np.rot90(np.rot90(pents[i]))
        if r==3:
            pent=np.rot90(np.rot90(np.rot90(pents[i])))
        if r==4:
            pent=np.flip(pents[i],1)
        if r==5:
            pent=np.flip(np.rot90(pents[i]),1)
        if r==6:
            pent=np.flip(np.rot90(np.rot90(pents[i])),1)
        if r==7:
            pent=np.flip(np.rot90(np.rot90(np.rot90(pents[i]))),1)
        
        output.append((pent,(var[1])))
        
    return output

    
    raise NotImplementedError