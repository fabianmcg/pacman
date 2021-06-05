#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from game import Agent
from agentRewards import *
from agentUtil import getActions
import time
import math
import numpy as np


class PacmanState:
    def __init__(self, state, action, reward, isTerminal, validActions=None):
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
        self.metrics = {
            "games": [],
            "totalActions": 0,
            "totalTime": 0,
        }
        self.actionIt = 0
        self.episodeIt = 0
        self.epsilon = float(epsilon)
        self.finalTrainingEpsilon = float(finalTrainingEpsilon)
        self.finalEpsilon = float(finalEpsilon)
        self.printSteps = int(printSteps)
        self.numTraining = int(numTraining)
        self.rewards = Rewards(**kwargs)
        self.random = np.random.default_rng(int(kwargs["seed"])) if "seed" in kwargs else np.random.default_rng(12345)
        self.epsilonStep = (self.epsilon - self.finalTrainingEpsilon) / (
            self.numTraining if self.numTraining > 0 else 1
        )
        self.epsilonArray = [self.epsilon, 1 - self.epsilon]
        self.startTime = None
        self.startGameTime = None
        self.stopTrainingTime = time.perf_counter()
        self.previousState = None
        self.parameters = {
            "epsilon": self.epsilon,
            "finalEpsilon": self.finalEpsilon,
            "finalTrainingEpsilon": self.finalTrainingEpsilon,
            "epsilonStep": self.epsilonStep,
            "numTraining": self.numTraining,
            "seed": int(kwargs["seed"]) if "seed" in kwargs else 12345,
        }

    def agentInit(self, gameState):
        pass

    def beginGame(self, gameState):
        pass

    def endGame(self, gameState):
        pass

    def updateJson(self):
        pass

    def selectAction(self, gameState, actions, actionsIndexes):
        return actions[0]

    def print(self, force=False):
        if force or (self.printSteps > 0 and ((self.episodeIt + 1) % self.printSteps) == 0):
            its = self.episodeIt if self.episodeIt < self.numTraining else (self.episodeIt - self.numTraining)
            its = 1 if its == 0 else its
            meanIts = min(its, 100)
            rewardScores = np.array([k[0] for k in self.metrics["games"]])
            scores = np.array([k[1] for k in self.metrics["games"]])
            wins = np.array([k[2] for k in self.metrics["games"]])
            actions = np.array([k[3] for k in self.metrics["games"]])
            print("Episode: {}".format(self.episodeIt + 1))
            print("\t{: <20}{:}".format("Total wins:", np.sum(wins)))
            print("\t{: <20}{:0.2f}".format("Total time:", self.metrics["totalTime"]))
            print("\t{: <20}{}".format("Total actions:", self.metrics["totalActions"]))
            print("\t{: <20}{:0.2f}".format("Win rate:", np.mean(wins[-meanIts:]) * 100.0))
            print("\t{: <20}{:0.2f}".format("Sup score:", np.amax(scores)))
            print("\t{: <20}{:0.2f}".format("Max score:", np.amax(scores[-meanIts:])))
            print("\t{: <20}{:0.2f}".format("Mean score:", np.mean(scores[-meanIts:])))
            print("\t{: <20}{:0.2f}".format("Mean actions:", np.mean(actions[-meanIts:])))
            print("\t{: <20}{:0.2f}".format("Sup total reward:", np.amax(rewardScores)))
            print("\t{: <20}{:0.2f}".format("Max total reward:", np.amax(rewardScores[-meanIts:])))
            print("\t{: <20}{:0.2f}".format("Mean total reward:", np.mean(rewardScores[-meanIts:])))

    def collectMetrics(self, gameState):
        stopGameTime = time.perf_counter()
        self.metrics["games"].append(
            [
                self.rewards.score,
                gameState.getScore(),
                gameState.isWin(),
                self.actionIt,
                stopGameTime - self.startGameTime,
                self.episodeIt < self.numTraining,
            ]
        )
        self.metrics["totalActions"] += self.actionIt
        self.metrics["totalTime"] = stopGameTime - self.startTime

    def toJson(self):
        self.updateJson()
        return {"parameters": self.parameters, "metrics": self.metrics}

    def registerInitialState(self, gameState):
        self.actionIt = 0
        self.rewards.initial(gameState)
        if self.episodeIt == 0:
            self.agentInit(gameState)
            self.startTime = time.perf_counter()
        if self.episodeIt == self.numTraining:
            self.metrics["trainingActions"] = self.metrics["totalActions"]
            self.metrics["totalActions"] = 0
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
        if (self.episodeIt + 1) == self.numTraining:
            self.print(True)
        else:
            self.print()
        self.episodeIt += 1

    def getAction(self, gameState):
        actions, actionsIndexes = getActions(gameState)
        action = self.selectAction(gameState, actions, actionsIndexes)
        self.actionIt += 1
        return action
