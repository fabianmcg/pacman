#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Rewards:
    def __init__(self, rewardFuntion=None, **kwargs):
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
        return score / 10.0

    def __call__(self, state):
        return self.reward(state)

 
