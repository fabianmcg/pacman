#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PacmanAgent import PacmanAgent, PacmanState
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

    def endGame(self, gameState):
        self.learn(tuple(gameStateVector(gameState)), gameState)

    def initAll(self, state, numActions):
        if state not in self.Pi:
            self.Q[state] = np.full(numActions, 0.0)
            self.Pi[state] = np.full(numActions, 1.0 / numActions)

    def learn(self, state, gameState):
        previousState = self.previousState.state
        previousAction = self.previousState.action
        numActions = len(self.previousState.validActions)
        reward = self.rewards(gameState)
        if state in self.Q:
            reward += self.gamma * np.amax(self.Q[state])
        Q = self.Q[previousState]
        Q[previousAction] = (1 - self.alpha) * Q[previousAction] + self.alpha * reward
        Pi = self.Pi[previousState]
        Pi[previousAction] += self.delta if previousAction == np.argmax(Q) else -self.delta / (numActions - 1)
        Pi = np.maximum(Pi, np.full(numActions, 0.0))
        self.Pi[previousState] = Pi / np.sum(Pi)

    def selectAction(self, gameState, actions, actionsIndexes):
        state = tuple(gameStateVector(gameState))
        self.initAll(state, len(actions))
        if self.previousState != None:
            self.learn(state, gameState)
        randomAction = self.random.choice([True, False], p=self.epsilonArray)
        if randomAction:
            action = self.random.integers(0, len(actions))
        else:
            Pi = self.Pi[state]
            action = self.random.choice(list(range(len(actionsIndexes))), p=Pi)
        self.previousState = PacmanState(state, action, 0, False, actionsIndexes)
        return actions[action]
