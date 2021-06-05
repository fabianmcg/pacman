#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from game import Agent
from agentRewards import *
from agentUtil import getActions
import time
import math
import numpy as np

class PacmanState:
    def __init__(self, state, action, reward, isTerminal, validActions = None):
        self.state = state
        self.action = action
        self.reward = reward
        self.isTerminal = isTerminal
        self.validActions = validActions

class PacmanAgent(Agent):
    def __init__(
        self,
        epsilon=1.0,
        printSteps=10,
        numTraining=0,
        finalTrainingEpsilon=0.1,
        finalEpsilon=0.005,
        **kwargs,
    ):
        self.index = 0
        self.metrics = {"meanScore": 0, "maxScore": -math.inf, "wins": 0, "totalActions": 0, "meanGameTime": 0, "totalTime": 0}
        self.actionIt = 0
        self.episodeIt = 0
        self.epsilon = float(epsilon)
        self.finalTrainingEpsilon = float(finalTrainingEpsilon)
        self.finalEpsilon = float(finalEpsilon)
        self.printSteps = int(printSteps)
        self.numTraining = int(numTraining)
        self.rewards = Rewards(**kwargs)
        self.random = np.random.default_rng(int(kwargs["seed"])) if "seed" in kwargs else np.random.default_rng(12345)
        self.epsilonStep = (self.epsilon - self.finalTrainingEpsilon) / (self.numTraining if self.numTraining > 0 else 1)
        self.epsilonArray = [self.epsilon, 1 - self.epsilon]
        self.startTime = None
        self.startGameTime = None
        self.stopTrainingTime = time.perf_counter()
        self.previousState = None

    def agentInit(self, gameState):
        pass

    def beginGame(self, gameState):
        pass

    def endGame(self, gameState):
        pass

    def selectAction(self, gameState, actions, actionsIndexes):
        return actions[0]

    def print(self, force=False):
        if force or (self.printSteps > 0 and (self.episodeIt % self.printSteps) == 0):
            its = self.episodeIt if self.episodeIt < self.numTraining else (self.episodeIt - self.numTraining)
            its = 1 if its == 0 else its
            print(
                "Episode: {}\n\t{: <20}{}\n\t{: <20}{:0.2f}\n\t{: <20}{}\n\t{: <20}{}\n\t{: <20}{}\n\t{: <20}{:0.2f}\n\t{: <20}{:0.2f}\n\t{: <20}{:0.2f}\n\t{: <20}{:0.2f}".format(
                    self.episodeIt,
                    "Total games:", its,
                    "Total time:", self.metrics["totalTime"],
                    "Total wins:", self.metrics["wins"],
                    "Total actions:", self.metrics["totalActions"],
                    "Is training:", self.episodeIt < self.numTraining,
                    "Max score:", self.metrics["maxScore"],
                    "Mean time:", self.metrics["meanGameTime"] / its,
                    "Mean score:", self.metrics["meanScore"] / its,
                    "Mean actions:", self.metrics["totalActions"] / its,
                )
            )

    def collectMetrics(self, gameState):
        stopGameTime = time.perf_counter()
        self.metrics["meanGameTime"] += stopGameTime - self.startGameTime
        self.metrics["meanScore"] += gameState.getScore()
        self.metrics["maxScore"] = max(gameState.getScore(), self.metrics["maxScore"])
        self.metrics["totalActions"] += self.actionIt
        self.metrics["totalTime"] = stopGameTime - self.startTime
        self.metrics["wins"] += gameState.isWin()

    def registerInitialState(self, gameState):
        self.actionIt = 0
        self.rewards.initial(gameState)
        if self.episodeIt == 0:
            self.agentInit(gameState)
            self.startTime = time.perf_counter()
        if self.episodeIt == self.numTraining:
            self.metrics["meanGameTime"] = 0
            self.metrics["meanScore"] = 0
            self.metrics["maxScore"] =  -math.inf
            self.metrics["totalActions"] = 0
            self.metrics["wins"] = 0
            self.epsilon = self.finalEpsilon
            self.epsilonArray = [self.epsilon, 1 - self.epsilon]
        self.startGameTime = time.perf_counter()
        self.beginGame(gameState)

    def final(self, gameState):
        self.endGame(gameState)
        self.previousState = None
        if self.episodeIt < self.numTraining:
            self.epsilon -= self.epsilonStep
            self.epsilonArray = [self.epsilon, 1 - self.epsilon]
        self.collectMetrics(gameState)
        if self.episodeIt == self.numTraining:
            self.print(True)
        else:
            self.print()
        self.episodeIt += 1

    def getAction(self, gameState):
        actions, actionsIndexes = getActions(gameState)
        action = self.selectAction(gameState, actions, actionsIndexes)
        self.actionIt += 1
        return action
