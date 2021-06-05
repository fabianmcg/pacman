#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import ma
from game import Directions

DIR2CODE = {Directions.STOP: 0, Directions.EAST: 1, Directions.NORTH: 2, Directions.WEST: 3, Directions.SOUTH: 4}
DIRECTIONS = [Directions.STOP, Directions.EAST, Directions.NORTH, Directions.WEST, Directions.SOUTH]


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


def ghostIndex(i, j=0, k=0, l=0):
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
    return (x, y, agent.scaredTimer)


def gameStateMatrixFull(gameState):
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
    grid = grid[1:-1, 1:-1] / 355.0
    return grid[..., np.newaxis]


def gameStateMatrixBasic(gameState):
    walls = np.array(gameState.getWalls().data, dtype=np.int16)
    food = np.array(gameState.getFood().data, dtype=np.int16)
    grid = 2 * food + walls
    for capsule in gameState.data.capsules:
        grid[getPositionTuple(capsule)] = 3
    grid[getPositionTuple(gameState.getPacmanPosition())] = 4
    for i, ghost in enumerate(gameState.getGhostStates()):
        grid[getPositionTuple(ghost.getPosition())] += (2 - (ghost.scaredTimer > 0)) * 5 * (3 ** i)
    grid = grid / (3 ** len(gameState.getGhostStates()))
    return grid[..., np.newaxis]


gameStateTensor = gameStateMatrixBasic


def gameStateVectorTuple(gameState):
    food = gameState.getFood().packBits()
    capsules = tuple([getPositionTuple(capsule) for capsule in gameState.data.capsules])
    pacman = getPositionTuple(gameState.getPacmanPosition())
    ghosts = tuple([serializeGhostState(ghost) for ghost in gameState.getGhostStates()])
    return pacman + ghosts + food[2:] + capsules


gameStateVector = gameStateVectorTuple
