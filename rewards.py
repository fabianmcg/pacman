#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from util import manhattanDistance


class Rewards:
    def __init__(self, rewardFuntion=None, **kwargs):
        self.score = 0
        self.food = 0
        self.capsules = 0
        self.currentReward = 0
        self.it = 0

    def initial(self, gameState):
        self.score = 0
        self.currentReward = 0
        self.food = gameState.getNumFood()
        self.capsules = len(gameState.getCapsules())
        self.it = 0

    def computeRewardI(self, gameState):
        if self.it == 0:
            return 0
        food = gameState.getNumFood()
        capsules = len(gameState.getCapsules())
        # pacmanPosition = gameState.getPacmanPosition()
        # nearestGhost = min(
        #     [manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in gameState.getGhostStates()]
        # )
        score = (
            256.0 * gameState.isWin()
            - 256.0 * gameState.isLose()
            + 32.0 * (self.food != food)
            + 64 * (self.capsules != capsules)
            - 1
        )
        self.currentReward = score / 256.0
        self.score += score
        self.food = food
        self.capsules = capsules
        self.it += 1
        return self.currentReward

    def computeRewardII(self, gameState):
        score = gameState.getScore() - self.score
        self.score = gameState.getScore()
        return score / 10.0

    def computeReward(self, state):
        return self.computeRewardI(state)

    def __call__(self):
        return self.currentReward
