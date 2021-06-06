#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from game import Agent, Directions
from rewards import *
from agentUtil import DIR2CODE, getActions, DIRECTIONS
import time
import numpy as np
from ast import literal_eval


class PacmanState:
    def __init__(self, state, reward, isTerminal, validActionsIndexes=None, action=None, validActions=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.isTerminal = isTerminal
        self.validActions = validActions
        self.validActionsIndexes = validActionsIndexes


class PacmanAgent(Agent):
    def __init__(
        self,
        epsilon=1.0,
        printSteps=10,
        numExplore=0,
        numTraining=0,
        finalEpsilon=0.005,
        finalTrainingEpsilon=0.1,
        noStopAction=None,
        sameActionPolicy = 0,
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
        self.printSteps = int(printSteps)
        self.numExplore = int(numExplore)
        self.numTraining = int(numTraining)
        self.finalEpsilon = float(finalEpsilon)
        self.finalTrainingEpsilon = float(finalTrainingEpsilon)
        self.noStopAction = True if noStopAction == None else literal_eval(noStopAction)
        self.sameActionPolicy = int(sameActionPolicy)
        self.rewards = Rewards(**kwargs)
        self.random = np.random.default_rng(int(kwargs["seed"])) if "seed" in kwargs else np.random.default_rng(12345)
        self.epsilonStep = (self.epsilon - self.finalTrainingEpsilon) / (
            (self.numTraining - self.numExplore) if self.numTraining > 0 else 1
        )
        self.epsilonArray = [self.epsilon, 1 - self.epsilon]
        self.startTime = None
        self.startGameTime = None
        self.previousState = None
        self.parameters = {
            "numTraining": self.numTraining,
            "numExplore": self.numExplore,
            "epsilon": self.epsilon,
            "finalEpsilon": self.finalEpsilon,
            "finalTrainingEpsilon": self.finalTrainingEpsilon,
            "epsilonStep": self.epsilonStep,
            "noStopAction": self.noStopAction,
            "sameActionPolicy": self.sameActionPolicy,
            "seed": int(kwargs["seed"]) if "seed" in kwargs else 12345,
        }

    def updateJson(self):
        pass

    def agentInit(self, gameState):
        pass

    def getState(self, gameState):
        return None

    def initState(self, agentState):
        pass

    def learn(self, agentState, isTerminal):
        pass

    def selectAction(self, agentState):
        return Directions.STOP

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

    def final(self, gameState):
        self.rewards.computeReward(gameState)
        state = PacmanState(
            state=self.getState(gameState),
            reward=self.rewards(),
            isTerminal=True,
            validActionsIndexes=[0],
            action=self.previousState.action
        )
        self.initState(state)
        if self.previousState != None and self.episodeIt >= self.numExplore:
            self.learn(state, True)
        self.previousState = None
        if self.episodeIt < self.numTraining and self.episodeIt >= self.numExplore:
            self.epsilon -= self.epsilonStep
            self.epsilonArray = [self.epsilon, 1 - self.epsilon]
        self.collectMetrics(gameState)
        if (self.episodeIt + 1) == self.numTraining:
            self.print(True)
        else:
            self.print()
        self.episodeIt += 1

    def observationFunction(self, gameState):
        self.rewards.computeReward(gameState)
        actions, actionsIndexes = getActions(gameState)
        if self.noStopAction:
            actions.remove(Directions.STOP)
            actionsIndexes.remove(DIR2CODE[Directions.STOP])
        state = PacmanState(
            state=self.getState(gameState),
            reward=self.rewards(),
            isTerminal=False,
            validActionsIndexes=actionsIndexes,
            validActions=actions,
            action=self.previousState.action if self.previousState != None else DIR2CODE[Directions.STOP], 
        )
        self.initState(state)
        if self.previousState != None and self.episodeIt >= self.numExplore:
            self.learn(state, False)
        return state

    def getAction(self, state):
        actions = state.validActions
        if (self.sameActionPolicy > 1) and ((self.actionIt % self.sameActionPolicy) != 0):
            action =  DIRECTIONS[self.previousState.action]
        elif self.epsilon > 0.0 and self.random.choice([True, False], p=self.epsilonArray):
            action = actions[self.random.integers(0, len(actions))]
        else:
            action = self.selectAction(state)
        self.previousState = PacmanState(
            state=state.state,
            reward=self.rewards(),
            isTerminal=False,
            validActionsIndexes=state.validActionsIndexes,
            action=DIR2CODE[action],
        )
        self.actionIt += 1
        if action not in actions:
            action = Directions.STOP
        return action
