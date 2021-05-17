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

class Rewards:
    def __init__(self):
        self.score = 0
    
    def reward(self, state):
        score = state.getScore() - self.score
        self.score = state.getScore()
        return score
    
    def reset(self):
        self.score = 0

class QState:
    def __init__(self, s, a, na):
        self.s = s
        self.a = a
        self.na = na

class PHCAgent(Agent):
    def __init__(self, a = 0.5, g = 0.75, d = 0.5, dl = 1., numTraining = 1):
        self.index = 0 # Pacman is always agent index 0
        self.gamma = float(g)
        self.alpha = float(a)
        self.delta = float(d)
        self.num_training = int(numTraining)
        self.it = 0
        self.Q = dict()
        self.pi = dict()
        self.rnd = np.random.default_rng(12345)
        self.last = None
        self.scoring_fn = Rewards()
    
    def final(self, state):
        if self.it < self.num_training:
            self.learn(state.serialize(), state.getScore())
        self.last = None
        self.scoring_fn.reset()
        self.it += 1

    def get_Qpi(self, state, n):
        if state in self.Q :
            return self.Q[state], self.pi[state]
        else:
            Q = self.Q[state] = np.full(n, 0.)
            pi = self.pi[state] = np.full(n, 1. / n)
            return Q, pi
    
    def learn(self, sp, reward):
        Q = self.Q[self.last.s]
        pi = self.pi[self.last.s]
        if sp in self.Q:
            reward += self.gamma * np.amax(self.Q[sp])
        Q[self.last.a] = (1 - self.alpha) * Q[self.last.a] + self.alpha * reward
        pi[self.last.a] += self.delta if  self.last.a == np.argmax(Q) else -self.delta / (self.last.na - 1)
        pi = np.maximum(pi, np.full(self.last.na, 0.))
        self.pi[self.last.s] = pi / np.sum(pi)

    def getAction(self, state):
        actions = state.getLegalActions()
        s = state.serialize()
        Q, pi = self.get_Qpi(s, len(actions))
        if self.it < self.num_training:
            action = self.rnd.choice(actions, p=pi)
            if self.last != None:
                self.learn(s, self.scoring_fn.reward(state))
        else:
            action = actions[np.argmax(Q)]
        self.last = QState(s, actions.index(action), len(actions))
        return action

class WPHCAgent(Agent):
    def __init__(self, a = 0.1, g = 0.75, d = 0.25, dl = 0.75, numTraining = 1):
        self.index = 0 # Pacman is always agent index 0
        self.gamma = float(g)
        self.alpha = float(a)
        self.delta_w = float(d)
        self.delta_l = float(dl)
        self.num_training = int(numTraining)
        self.it = 0
        self.Q = dict()
        self.C = dict()
        self.pi = dict()
        self.pih = dict()
        self.rnd = np.random.default_rng(seed=42)
        self.last = None
        self.scoring_fn = Rewards()
    
    def final(self, state):
        if self.it < self.num_training:
            self.learn(state.serialize(), state.getScore())
        self.last = None
        self.scoring_fn.reset()
        self.it += 1

    def get_Qpi(self, state, n):
        if state in self.Q :
            return self.Q[state], self.pi[state]
        else:
            Q = self.Q[state] = np.full(n, 0.)
            pi = self.pi[state] = np.full(n, 1. / n)
            self.C[state] = 0
            self.pih[state] = np.full(n, 1. / n)
            return Q, pi
    
    def learn(self, sp, reward):
        Q = self.Q[self.last.s]
        pi = self.pi[self.last.s]
        self.C[self.last.s] += 1
        C = self.C[self.last.s]
        pih = self.pih[self.last.s]
        pih = pih + (pi - pih) / C
        self.pih[self.last.s] = pih
        if sp in self.Q:
            reward += self.gamma * np.amax(self.Q[sp])
        Q[self.last.a] = (1 - self.alpha) * Q[self.last.a] + self.alpha * reward
        piQ = np.dot(pi, Q)
        pihQ = np.dot(pih, Q)
        delta = self.delta_w if piQ > pihQ else self.delta_l
        pi[self.last.a] += delta if  self.last.a == np.argmax(Q) else -delta / (self.last.na - 1)
        pi = np.maximum(pi, np.full(self.last.na, 0.))
        self.pi[self.last.s] = pi / np.sum(pi)

    def getAction(self, state):
        actions = state.getLegalActions()
        s = state.serialize()
        Q, pi = self.get_Qpi(s, len(actions))
        if self.it < self.num_training:
            action = self.rnd.choice(actions, p=pi)
            if self.last != None:
                self.learn(s, self.scoring_fn.reward(state))
        else:
            action = actions[np.argmax(Q)]
        self.last = QState(s, actions.index(action), len(actions))
        return action
