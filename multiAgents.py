# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import numpy as np
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class QAgent(Agent):
    def __init__(self, alpha = 0.5, gamma = 0.75, delta = 0.5, numTraining = 1):
        self.index = 0 # Pacman is always agent index 0
        self.gamma = gamma
        self.alpha = alpha
        self.delta = delta
        self.Q = dict()
        self.pi = dict()
        self.rnd = np.random.default_rng(12345)
        self.score = 0
        self.last = [None, None, None, None, None, None, None]
    
    def serializeState(self, gameState):
        pos = gameState.getPacmanPosition()
        gpos = gameState.getGhostStates()
        food = gameState.getFood()
        return pos, food, gpos[0].getPosition()

    def final(self, state):
        self.Q[self.last[2]] = self.last[0]
        self.pi[self.last[2]] = self.last[1]
        self.update(self.last[2], self.last[3], self.last[4], state.getScore() - self.last[5], self.last[6])
        self.score = 0

    def giQpi(self, state, n):
        if state in self.Q :
            return self.Q[state], self.pi[state]
        else:
            Q = self.Q[state] = np.full(n, 0.)
            pi = self.pi[state] = np.full(n, 1. / n)
            return Q, pi
    
    def update(self, cst, nst, aid, s, n):
        Q = self.Q[cst]
        pi = self.pi[cst]
        if nst in self.Q:
            s += self.gamma * np.amax(self.Q[nst])
        Q[aid] = (1 - self.alpha) * Q[aid] + self.alpha * s
        pi[aid] += self.delta if  aid == np.argmax(Q) else -self.delta / (n - 1)
        pi = np.maximum(pi, np.full(n, 0.))
        self.pi[cst] = pi / np.sum(pi)

    def getAction(self, gameState):
        legalMoves = gameState.getLegalActions()
        cst = self.serializeState(gameState)
        Q, pi = self.giQpi(cst, len(legalMoves))
        action = self.rnd.choice(legalMoves, p=pi)
        successorGameState = gameState.generatePacmanSuccessor(action)
        nst = self.serializeState(successorGameState)
        aid = legalMoves.index(action)
        self.last[0] = np.copy(Q)
        self.last[1] = np.copy(pi)
        self.last[2] = cst
        self.last[3] = nst
        self.last[4] = aid
        self.last[5] = self.score
        self.last[6] = len(legalMoves)
        self.update(cst, nst, aid, successorGameState.getScore() - self.score, len(legalMoves))
        self.score = successorGameState.getScore()
        return action