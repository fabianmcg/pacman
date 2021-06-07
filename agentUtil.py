#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import ma
from game import Directions

DIR2CODE = {Directions.EAST: 0, Directions.NORTH: 1, Directions.WEST: 2, Directions.SOUTH: 3, Directions.STOP: 4}
DIRECTIONS = [Directions.EAST, Directions.NORTH, Directions.WEST, Directions.SOUTH, Directions.STOP]


def getActions(gameState):
    actions = gameState.getLegalActions()
    return actions, [DIR2CODE[a] for a in actions]


def getPositionTuple(pos):
    return (int(pos[0] + 0.5), int(pos[1] + 0.5))


def getVectorPosition(agent):
    x, y = agent.getPosition()
    d = DIR2CODE[agent.getDirection()]
    return (x, y, d)


def serializeGhostState(agent):
    x, y = getPositionTuple(agent.getPosition())
    return (x, y, agent.scaredTimer)


from scipy import ndimage


def gameStateMatrixScaled(gameState):
    walls = np.array(gameState.getWalls().data, dtype=np.int16)
    food = np.array(gameState.getFood().data, dtype=np.int16)
    grid = 2 * food + walls
    for capsule in gameState.data.capsules:
        grid[getPositionTuple(capsule)] = 3
    grid[getPositionTuple(gameState.getPacmanPosition())] = 4
    for i, ghost in enumerate(gameState.getGhostStates()):
        grid[getPositionTuple(ghost.getPosition())] += (2 - (ghost.scaredTimer > 0)) * 5 * (3 ** i)
    x = 32 / grid.shape[0]
    y = 32 / grid.shape[1]
    grid = ndimage.zoom(grid, (x, y), order=0, grid_mode=True, mode="nearest")
    grid = grid / ((3 ** len(gameState.getGhostStates())) * 5)
    return grid[..., np.newaxis]


def gameStateMatrix(gameState):
    walls = np.array(gameState.getWalls().data, dtype=np.int16)
    food = np.array(gameState.getFood().data, dtype=np.int16)
    grid = 2 * food + walls
    for capsule in gameState.data.capsules:
        grid[getPositionTuple(capsule)] = 3
    grid[getPositionTuple(gameState.getPacmanPosition())] = 4
    for ghost in gameState.getGhostStates():
        position = getPositionTuple(ghost.getPosition())
        grid[position] = max(6 - (ghost.scaredTimer > 0), grid[position])
    return grid.T / 6.0


def gameStateTensorSimple(gameState):
    grid = gameStateMatrix(gameState)
    return grid[..., np.newaxis]


gameStateTensor = gameStateTensorSimple


def gameStateVectorTuple(gameState):
    food = gameState.getFood().packBits()
    capsules = tuple([getPositionTuple(capsule) for capsule in gameState.data.capsules])
    pacman = getPositionTuple(gameState.getPacmanPosition())
    ghosts = tuple([serializeGhostState(ghost) for ghost in gameState.getGhostStates()])
    return pacman + ghosts + food[2:] + capsules


gameStateVector = gameStateVectorTuple


__image_number = 0
def saveMatrixAsImage(matrix, name = None):
    from PIL import Image
    global __image_number
    if name == None:
        name = "{:04d}.png".format(__image_number)
        __image_number += 1
    Image.fromarray(np.uint8(matrix * 255.0), 'L').save(name)
