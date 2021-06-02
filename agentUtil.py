#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import ma
from game import Directions

DIR2CODE = {
        Directions.EAST:  0,
        Directions.NORTH: 1,
        Directions.WEST:  2,
        Directions.SOUTH: 3,
        Directions.STOP:  4
        }

def sumToN(n):
    return int((n * (n + 1)) / 2)

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
        return triangular2TIndex(i, j, 8)
    elif l == 0:
        return triangular3TIndex(i, j, k, 8)
    return triangular4TIndex(i, j, k, l, 8)

def getPositionTuple(pos):
    return (int(pos[0]), int(pos[1]))

def gameStateVector(gameState):
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
            ghosts[(x, y)] = []
        ghosts[(x, y)].append((ghost.scaredTimer > 0) * 4 + DIR2CODE[ghost.getDirection()])
    for x, y in ghosts:
        grid[x, y] += ghostIndex(*ghosts[(x, y)]) * 5
    timers = [ghost.scaredTimer for ghost in gameState.getGhostStates()]
    timers.append(0)
    return np.append(grid[1:-1, 1:-1].flatten(), max(timers))

class Rewards:
    def __init__(self, rewardFuntion = None, **kwargs):
        self.score = 0
    
    def reward(self, state):
        score = state.getScore() - self.score
        self.score = state.getScore()
        return score
        
    def reset(self):
        self.score = 0
    
    def __call__(self, state):
        return self.reward(state)

class QState:
    def __init__(self, state, action, numActions):
        self.state = state
        self.action = action
        self.numActions = numActions