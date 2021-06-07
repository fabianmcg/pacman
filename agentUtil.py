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
    capsules = np.full(walls.shape, 0)
    pacman = np.full(walls.shape, 0)
    ghosts = np.full(walls.shape, 0)
    scaredGhosts = np.full(walls.shape, 0)
    for capsule in gameState.data.capsules:
        capsules[getPositionTuple(capsule)] = 1
    pacman[getPositionTuple(gameState.getPacmanPosition())] = 1
    for ghost in gameState.getGhostStates():
        position = getPositionTuple(ghost.getPosition())
        if (ghost.scaredTimer <= 0):
            ghosts[position] = 1
        else:
            scaredGhosts[position] = 1
    grid = np.stack(tuple([walls, pacman, ghosts, scaredGhosts, food, capsules]), axis=-1)
    return grid


gameStateTensor = gameStateMatrix


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
