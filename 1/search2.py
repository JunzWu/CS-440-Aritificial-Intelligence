# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Rahul Kunji (rahulsk2@illinois.edu) on 01/16/2019

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrev0ised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)

import queue
import heapq

class DisjointSet:
    def __init__(self, vertices):
        self.vertices=vertices[:]
        self.parent={}
        for n in self.vertices:
            self.parent[n]=n
    
    def find(self,item):
        if self.parent[item] == item:
            return item
        else:
            return self.find(self.parent[item])
    
    def union(self,set1,set2):
        root1 = self.find(set1)
        root2 = self.find(set2)
        self.parent[root1] = root2
            
def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
    }.get(searchMethod)(maze)

def bfs_(maze, start, dest):
    q=queue.Queue()
    q.put([start,start])
    explored=set()
    parent={}
    path_=[]
    while not q.empty():
        curr=q.get()
        if curr[0] in explored:
            continue
        explored.add(curr[0])
        parent.update({curr[0]:curr[1]})
        if curr[0]==dest:
            pathback=curr[0]
            path_.append(pathback)
            parentback=parent.get(pathback)
            while parentback != start:
                path_.append(parentback)
                parentback=parent.get(parentback) 
            return len(path_)+1
 
        neighbors=maze.getNeighbors(curr[0][0],curr[0][1])
        for p in neighbors:
                q.put([p,curr[0]])
                

def bfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    start=maze.getStart()
    starttemp=start
    q=queue.Queue()
    q.put([start,start])
    explored=set()
    parent={}
    dest=maze.getObjectives()
    path=[]
    path_=[]
    states=1
    while not q.empty():
        curr=q.get()
        if curr[0] in explored:
            continue
        explored.add(curr[0])
        parent.update({curr[0]:curr[1]})
        states=states+1
        for d in dest:
            if curr[0]==d:
                pathback=curr[0]
                path_.append(pathback)
                parentback=parent.get(pathback)
                while parentback != starttemp:
                    path_.append(parentback)
                    parentback=parent.get(parentback) 
                path_.reverse()
                path+=path_
                path_.clear()
                dest.remove(d)
                if len(dest)==0:
                    break      
                else:
                    explored.clear()
                    explored.add(curr[0])
                    parent.clear()
                    q.queue.clear()
                    starttemp=curr[0]
            
        neighbors=maze.getNeighbors(curr[0][0],curr[0][1])
        for p in neighbors:
                q.put([p,curr[0]])
                
        
    path.insert(0,start)
    
    return path,states


def dfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    start=maze.getStart()
    s=[]
    s.append([start,start])
    explored=set()
    parent={}
    dest=maze.getObjectives()
    path=[]
    while  len(s)!=0:
        curr=s.pop()
        if curr[0] in explored:
            continue
        explored.add(curr[0])
        parent.update({curr[0]:curr[1]})
        
        if curr[0]==dest[0]:
            pathback=curr[0]
            path.append(pathback)
            parentback=parent.get(pathback)
            while parentback != start:
                path.append(parentback)
                parentback=parent.get(parentback) 
            break
            
        neighbors=maze.getNeighbors(curr[0][0],curr[0][1])
        for p in neighbors:
                s.append([p,curr[0]])
                
        
    path.append(start)
    path.reverse()
    
    return path, len(explored)+1

def greedy_(maze, start, dest):
    heap=[]
    heapq.heappush(heap,(abs(dest[0]-start[0])+abs(dest[1]-start[1]),start,start))
    explored=set()
    parent={}
    path=[]
    while  len(heap)!=0:
        curr=heapq.heappop(heap)
        if curr[1] in explored:
            continue
        explored.add(curr[1])
        parent.update({curr[1]:curr[2]})
        if curr[1]==dest:
            pathback=curr[1]
            path.append(pathback)
            parentback=parent.get(pathback)
            while parentback != start:
                path.append(parentback)
                parentback=parent.get(parentback) 
            return len(path)+1
            
        neighbors=maze.getNeighbors(curr[1][0],curr[1][1])
        for p in neighbors:
                if p not in explored:
                    dist=abs(dest[0]-p[0])+abs(dest[1]-p[1])
                    heapq.heappush(heap,(dist,p,curr[1]))
                

def greedy(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    start=maze.getStart()
    starttemp=start
    dest=maze.getObjectives()
    heap=[]
    heapq.heappush(heap,(minhn(dest,start),start,start))
    explored=set()
    parent={}
    path=[]
    path_=[]
    states=1
    while  len(heap)!=0:
        curr=heapq.heappop(heap)
        if curr[1] in explored:
            continue
        explored.add(curr[1])
        parent.update({curr[1]:curr[2]})
        states=states+1
        for d in dest:
            if curr[1]==d:
                pathback=curr[1]
                path_.append(pathback)
                parentback=parent.get(pathback)
                while parentback != starttemp:
                    path_.append(parentback)
                    parentback=parent.get(parentback) 
                path_.reverse()
                path+=path_
                path_.clear()
                dest.remove(d)
                if len(dest)==0:
                    break      
                else:
                    explored.clear()
                    explored.add(curr[1])
                    parent.clear()
                    heap.clear()
                    starttemp=curr[1]
                
        if len(dest)==0:
            break;
            
        neighbors=maze.getNeighbors(curr[1][0],curr[1][1])
        for p in neighbors:
                if p not in explored:
                    dist=minhn(dest,p)
                    heapq.heappush(heap,(dist,p,curr[1]))
                
        
    path.insert(0,start)
    
    return path, states


def minhn(dest,curr):
    hn=[]
    for i in dest:
      hn.insert(0,abs(i[0]-curr[0])+abs(i[1]-curr[1]))
    return min(hn)

def MST(maze,dest_,curr):
    heap=[]
    dest=dest_[:]
    dest.insert(0,curr)
    myset=DisjointSet(dest)
    for i in range(len(dest)-1):
        for j in range(i+1,len(dest)): 
            distance=abs(dest[i][0]-dest[i][0])+abs(dest[j][1]-dest[i][1])
            heapq.heappush(heap,(distance,dest[i],dest[j]))
    out=0
    while len(heap)!=0:
        curr=heapq.heappop(heap)
        if myset.find(curr[1])!=myset.find(curr[2]):
            myset.union(curr[1],curr[2])
            out+=curr[0]
    return out
      
def astar(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    start=maze.getStart()
    dest=maze.getObjectives()
    heap=[]
    starttemp=start
    heapq.heappush(heap,(minhn(dest,start),0,start,start))  # (g(n)+h(n), g(n), self, parent)
    explored=set()
    parent={}
    gh={}
    path=[]
    path_=[]
    states=1
    while  len(heap)!=0:
        curr=heapq.heappop(heap)
        if curr[2] in explored:
            if curr[0]<gh.get(curr[2]):
                gh.update({curr[2]:curr[0]})
                parent.update({curr[2]:curr[3]})
            continue
        explored.add(curr[2])
        states=states+1
        gh.update({curr[2]:curr[0]})
        parent.update({curr[2]:curr[3]})
        
        for d in dest:
            if curr[2]==d:
                pathback=curr[2]
                path_.append(pathback)
                parentback=parent.get(pathback)
                while parentback != starttemp:
                    path_.append(parentback)
                    parentback=parent.get(parentback) 
                path_.reverse()
                path+=path_
                path_.clear()
                dest.remove(d)
                if len(dest)==0:
                    break      
                else:
                    explored.clear()
                    explored.add(curr[2])
                    parent.clear()
                    heap.clear()
                    gh.clear()
                    gh.update({curr[2]:curr[0]})
                    starttemp=curr[2]
        if len(dest)==0:
            break
        neighbors=maze.getNeighbors(curr[2][0],curr[2][1])
        for p in neighbors:
            dist=minhn(dest,p)
            gn=curr[1]+1
            heapq.heappush(heap,(gn+dist,gn,p,curr[2]))
                
        
    path.insert(0,start)
    
    return path, states
