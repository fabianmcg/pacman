#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from util import manhattanDistance
from ast import literal_eval


class Rewards:
    def __init__(self, clipReward=None, **kwargs):
        self.score = 0
        self.food = 0
        self.capsules = 0
        self.currentReward = 0
        self.it = 0
        self.clipReward = False if clipReward == None else literal_eval(clipReward)
        self.normalizer = 501.0 if self.clipReward else 1.0

    def initial(self, gameState):
        self.score = 0
        self.currentReward = 0
        self.food = gameState.getNumFood()
        self.capsules = len(gameState.getCapsules())
        self.it = 0

    def computeRewardI(self, gameState):
        if self.it == 0:
            self.it += 1
            return 0
        food = gameState.getNumFood()
        capsules = len(gameState.getCapsules())
        # pacmanPosition = gameState.getPacmanPosition()
        # nearestGhost = min(
        #     [manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in gameState.getGhostStates()]
        # )
        score = (
            100 * gameState.isWin()
            - 500 * gameState.isLose()
            + 10 * (self.food != food)
            + 50 * (self.capsules != capsules)
            - 1
        )
        self.currentReward = score / self.normalizer
        self.score += score
        self.food = food
        self.capsules = capsules
        return self.currentReward

    def computeRewardII(self, gameState):
        score = gameState.getScore() - self.score
        self.score = gameState.getScore()
        return score / 10.0

    def computeReward(self, state):
        return self.computeRewardI(state)

    def __call__(self):
        return self.currentReward

    def toJson(self):
        return {"clipReward": self.clipReward}
