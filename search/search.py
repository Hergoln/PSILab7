# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import Queue, manhattanDistance, PriorityQueue, PriorityQueueWithFunction, Stack
from game import Actions, Directions
import copy
import heapq

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def uglyDepthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    
    State -> tuple: (x, y):
        Successors -> List:
        ListEl:
            tuple: (x, y)
            string: Direction -> game.Directions
            cost: int

            successor: coords, Direction, cost
            
    problem interface: 
        getStartState(self); 
        isGoalState(self, state);
        getSuccessors(self, state);
        getCostOfActions(self, actions);
    """
    from game import Directions
    from util import Stack
    
    to_look = Stack()
    out = Stack()
    visited = []
    current_state = (problem.getStartState(), Directions.STOP, 0)
    child = current_state
    not_visited = 0
    
    while True:
        if problem.isGoalState(current_state[0]):
            out.list.reverse()
            dirs = []
            while not out.isEmpty():
                pair = out.pop()
                dirs.append(Actions.vectorToDirection((pair[1][0][0] - pair[0][0][0], pair[1][0][1] - pair[0][0][1])))
            return dirs
        
        successors = problem.getSuccessors(current_state[0])
        for successor in successors:
            if successor[0] not in visited:
                to_look.push((current_state, successor))
                not_visited = not_visited + 1
        
        child = to_look.pop()
        temp = (None, None)
        if not_visited <= 0:
            while temp[0] != child[0]:
                temp = out.pop()
        
        not_visited = 0
        out.push(child)
        visited.append(current_state[0])
        current_state = child[1]
        
    ###
def depthFirstSearch(problem):
    # queue is queue of states
    queue = Stack()
    current_state = problem.getStartState()
    actions = []
    visited = []
    queue.push((current_state, actions))
    
    while queue:
        current_state, actions = queue.pop()
        if current_state not in visited:
            visited.append(current_state)
            if problem.isGoalState(current_state):
                return actions
            
            for successor in problem.getSuccessors(current_state):
                state, action, _ = successor
                newActions = actions + [action]
                queue.push((state, newActions))
                
    return []


def breadthFirstSearch(problem):
    # queue is queue of states
    queue = Queue()
    current_state = problem.getStartState()
    actions = []
    visited = []
    # queue of pairs -> (state, [actions])
    queue.push((current_state, actions))
    
    while queue:
        current_state, actions = queue.pop()
        #successor:
        #   coords: (x, y)
        #   Direction: game.Directions
        #   cost: int
        if current_state not in visited:
            visited.append(current_state)
            if problem.isGoalState(current_state):
                return actions
            
            for successor in problem.getSuccessors(current_state):
                state, action, _ = successor
                newActions = actions + [action]
                queue.push((state, newActions))
                
    return []
    
class PriorityQueueNetflixAdaptation(PriorityQueue):
    def  __init__(self):
        PriorityQueue.__init__(self)

    def pop(self):
        (priority, _, item) = heapq.heappop(self.heap)
        return (item, priority)

def uniformCostSearch(problem):
    queue = PriorityQueue()
    current_state = problem.getStartState()
    actions = []
    visited = []
    queue.push((current_state, actions), 0)
    
    while queue:
        current_state, actions = queue.pop()

        if current_state not in visited:
            visited.append(current_state)
            if problem.isGoalState(current_state):
                return actions
            
            for successor in problem.getSuccessors(current_state):
                state, action, cost = successor
                cost += problem.getCostOfActions(actions)
                newActions = actions + [action]
                queue.push((state, newActions), cost)
                
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def oldAStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # i is state = ((x, y), dir, cost)
    priorityFunc = lambda item: sum(i[2] for i in item) + heuristic(item[len(item) - 1][0], problem)
        
    # queue is queue of states
    queue = PriorityQueueWithFunction(priorityFunc)
    #pathsQueue = PriorityQueueWithFunction(priorityFunc)
    current_state = [] # array of arrays
    current_state.append((problem.getStartState(), Directions.STOP, 0))
    queue.push(current_state)
    while True:
        # current_state is an array of states
        current_state = queue.pop()
        #successor(state):
        #   coords: (x, y)
        #   Direction: game.Directions
        #   cost: int
        for successor in problem.getSuccessors(current_state[len(current_state) - 1][0]):
            state_copy = copy.deepcopy(current_state)
            if successor[0] not in list(map(lambda item: item[0], state_copy)):
                state_copy.append(successor)
                queue.push(state_copy)
            
            if problem.isGoalState(successor[0]):
                # from coord to dirs
                return list(map(lambda item: item[1], state_copy))

def aStarSearch(problem, heuristic=nullHeuristic):
    priorityFunc = lambda state: problem.getCostOfActions(state[1]) + heuristic(state[0], problem)
    queue = PriorityQueueWithFunction(priorityFunc)
    current_state = problem.getStartState()
    actions = []
    visited = []
    # state depends on problem now but actions list is independent thus priorityFunc should work on every problem
    # queue of pairs -> (state, [actions])
    queue.push((current_state, actions))
    
    while queue:
        current_state, actions = queue.pop()
        #successor:
        #   coords: (x, y)
        #   Direction: game.Directions
        #   cost: int
        if current_state not in visited:
            visited.append(current_state)
            if problem.isGoalState(current_state):
                return actions
            
            for successor in problem.getSuccessors(current_state):
                state, action, _ = successor
                newActions = actions + [action]
                queue.push((state, newActions))
                
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
