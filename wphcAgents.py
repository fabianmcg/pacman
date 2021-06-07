#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from game import Agent
from agentUtil import *
from phcAgents import PHCAgent


class WPHCAgent(PHCAgent):
    def __init__(self, delta=0.5, deltaLose=0.75, **kwargs):
        super().__init__(delta=delta, **kwargs)
        self.deltaLose = float(deltaLose)
        self.C = dict()
        self.PiAvg = dict()
        self.parameters.update(
            {
                "deltaLose": self.deltaLose,
            }
        )

    def initState(self, agentState):
        state = agentState.state
        if state not in self.Pi:
            numActions = len(agentState.validActionsIndexes)
            self.Q[state] = np.full(numActions, 0.0)
            self.Pi[state] = np.full(numActions, 1.0 / numActions)
            self.C[state] = 0
            self.PiAvg[state] = np.full(numActions, 1.0 / numActions)

    def learn(self, agentState, isTerminal):
        previousState = self.previousState.state
        previousAction = self.previousState.validActionsIndexes.index(self.previousState.action)
        numActions = len(self.previousState.validActionsIndexes)
        state = agentState.state
        reward = agentState.reward
        if state in self.Q:
            reward += self.gamma * np.amax(self.Q[state])
        Q = self.Q[previousState]
        Q[previousAction] = (1 - self.alpha) * Q[previousAction] + self.alpha * reward
        self.C[previousState] += 1
        Pi = self.Pi[previousState]
        PiAvg = self.PiAvg[previousState]
        PiAvg += (Pi - PiAvg) / self.C[previousState]
        delta = self.delta if np.dot(Pi, Q) > np.dot(PiAvg, Q) else self.deltaLose
        Pi[previousAction] += delta if previousAction == np.argmax(Q) else -delta / (numActions - 1)
        Pi = np.maximum(Pi, np.full(numActions, 0.0))
        self.Pi[previousState] = Pi / np.sum(Pi)
