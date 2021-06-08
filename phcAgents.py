#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from agent import PacmanAgent
from agentUtil import *


class PHCAgent(PacmanAgent):
    def __init__(self, alpha=0.25, gamma=0.75, delta=0.75, **kwargs):
        super().__init__(**kwargs)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.Q = dict()
        self.Pi = dict()
        self.parameters.update(
            {
                "gamma": self.gamma,
                "alpha": self.alpha,
                "delta": self.delta,
            }
        )

    def updateJson(self):
        self.parameters["numExploredStates"] = len(self.Q)

    def getState(self, gameState):
        return gameStateVector(gameState)

    def initState(self, agentState):
        state = agentState.state
        if state not in self.Pi:
            numActions = len(agentState.validActionsIndexes)
            self.Q[state] = np.full(numActions, 0.0)
            self.Pi[state] = np.full(numActions, 1.0 / numActions)

    def learn(self, agentState):
        previousState = self.previousState.state
        previousAction = self.previousState.validActionsIndexes.index(self.previousState.action)
        numActions = len(self.previousState.validActionsIndexes)
        state = agentState.state
        reward = agentState.reward
        if state in self.Q:
            reward += self.gamma * np.amax(self.Q[state])
        Q = self.Q[previousState]
        Q[previousAction] = (1 - self.alpha) * Q[previousAction] + self.alpha * reward
        Pi = self.Pi[previousState]
        Pi[previousAction] += self.delta if previousAction == np.argmax(Q) else -self.delta / (numActions - 1)
        Pi = np.maximum(Pi, np.full(numActions, 0.0))
        self.Pi[previousState] = Pi / np.sum(Pi)

    def selectAction(self, agentState):
        Pi = self.Pi[agentState.state]
        action = self.random.choice(list(range(len(agentState.validActions))), p=Pi)
        return agentState.validActions[action]
