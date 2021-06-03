#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import ma
from game import Directions
from util import manhattanDistance

DIR2CODE = {
        Directions.EAST:  0,
        Directions.NORTH: 1,
        Directions.WEST:  2,
        Directions.SOUTH: 3,
        Directions.STOP:  4
        }

def getActions(gameState):
    actions = gameState.getLegalActions()
    return actions, [DIR2CODE[a] for a in actions]

def sumSumToN(n):
    return int((n * (n + 1) * (n + 2)) / 6)

def sumSumSumToN(n):
    return int((n * (n + 1) * (n + 2) * (n + 3)) / 24)

def triangular2TIndex(i, j, n):    
    return int(i - j + ((2 * n - j + 1) * j) / 2)

def triangular3TIndex(i, j, k, n):
    return triangular2TIndex(i - k, j - k, n - k) + sumSumToN(n) - sumSumToN(n - k)

def triangular4TIndex(i, j, k, l, n):
    return triangular3TIndex(i - l, j - l, k - l, n - l) + sumSumSumToN(n) - sumSumSumToN(n - l)

def ghostIndex(i, j = 0, k = 0, l = 0):
    if l == 0 and k == 0 and j == 0:
        return i
    elif l == 0 and k == 0:
        return triangular2TIndex(i - 1, j, 7)
    elif l == 0:
        return triangular3TIndex(i - 2, j - 1, k, 6)
    return triangular4TIndex(i - 3, j - 2, k - 1, l, 5)

def getPositionTuple(pos):
    return (int(pos[0] + 0.5), int(pos[1] + 0.5))

def getVectorPosition(agent):
    x, y = agent.getPosition()
    d = DIR2CODE[agent.getDirection()]
    return (x, y, d)

def serializeGhostState(agent):
    x, y = getPositionTuple(agent.getPosition())
    d = DIR2CODE[agent.getDirection()]
    return (x, y, d, agent.scaredTimer)

def gameStateMatrix(gameState):
    walls = np.array(gameState.getWalls().data, dtype=np.int16)
    food = np.array(gameState.getFood().data, dtype=np.int16)
    grid = 2 * food + walls
    for capsule in gameState.data.capsules:
        grid[getPositionTuple(capsule)] = 3
    grid[getPositionTuple(gameState.getPacmanPosition())] = 4
    ghosts = {}
    for ghost in gameState.getGhostStates():
        x, y = getPositionTuple(ghost.getPosition())
        if (x, y) not in ghosts:
            ghosts[(x, y)] = set()
        ghosts[(x, y)].add((ghost.scaredTimer > 0) * 4 + DIR2CODE[ghost.getDirection()])
    for x, y in ghosts:
        grid[x, y] += (1 + ghostIndex(*list(ghosts[(x, y)]))) * 5
    grid = grid[1:-1, 1:-1] / 355.
    return grid[..., np.newaxis]

def gameStateMatrix2(gameState):
    walls = np.array(gameState.getWalls().data, dtype=np.int16)
    food = np.array(gameState.getFood().data, dtype=np.int16)
    grid = 2 * food + walls
    for capsule in gameState.data.capsules:
        grid[getPositionTuple(capsule)] = 3
    grid[getPositionTuple(gameState.getPacmanPosition())] = 4
    for ghost in gameState.getGhostStates():
        grid[getPositionTuple(ghost.getPosition())] += (1 + (ghost.scaredTimer > 0)) * 5
    grid = grid[1:-1, 1:-1] / 10.
    return grid[..., np.newaxis]

gameStateTensor = gameStateMatrix2

def gameStateVectorTuple(gameState):
    food = gameState.getFood().packBits()
    capsules = tuple([getPositionTuple(capsule) for capsule in gameState.data.capsules])
    pacman = getPositionTuple(gameState.getPacmanPosition())
    ghosts = tuple([serializeGhostState(ghost) for ghost in gameState.getGhostStates()])
    return pacman + ghosts + food[2:] + capsules

def gameStateVectorFromMatrix(gameState):
    state = gameStateMatrix(gameState).flatten()
    timers = [ghost.scaredTimer for ghost in gameState.getGhostStates()]
    timers.append(0)
    return np.append(state, max(timers))

gameStateVector = gameStateVectorTuple

class Rewards:
    def __init__(self, rewardFuntion = None, **kwargs):
        self.score = 0
        self.food = 0
        self.capsules = 0
    
    def initial(self, gameState):
        self.food = gameState.getNumFood()
        self.capsules = len(gameState.getCapsules())
        self.score = 0

    # def reward(self, gameState):
    #     pacmanPosition = gameState.getPacmanPosition()
    #     food = gameState.getNumFood()
    #     capsules = len(gameState.getCapsules())
    #     nearestGhost = min([manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in gameState.getGhostStates()])
    #     score = 500. * gameState.isWin() - 500. * gameState.isLose() + 20 * (self.food != food) - (nearestGhost < 3) * 5 - 1 + 100 * (self.capsules != capsules)
    #     self.score = score
    #     self.food = food
    #     self.capsules = capsules
    #     return score / 500.

    def reward(self, gameState):
        score = gameState.getScore() - self.score
        self.score = gameState.getScore()
        return score / 501.
        
    def reset(self):
        self.score = 0
        self.food = 0
        self.capsules = 0
    
    def __call__(self, state):
        return self.reward(state)

class QState:
    def __init__(self, state, action, numActions = 0, validActions = None):
        self.state = state
        self.action = action
        self.numActions = numActions
        self.validActions = validActions
