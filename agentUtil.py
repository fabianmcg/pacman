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


def serializeGhostState(agent):
    x, y = getPositionTuple(agent.getPosition())
    return (x, y, agent.scaredTimer)


def gameStateMatrix(gameState):
    walls = np.array(gameState.getWalls().data, dtype=np.int8)
    food = np.array(gameState.getFood().data, dtype=np.int8)
    capsules = np.full(walls.shape, 0, dtype=np.int8)
    pacman = np.full(walls.shape, 0, dtype=np.int8)
    ghosts = np.full(walls.shape, 0, dtype=np.int8)
    scaredGhosts = np.full(walls.shape, 0, dtype=np.int8)
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
