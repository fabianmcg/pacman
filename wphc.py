#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from game import Agent
from agentUtil import *
from phc import PHCAgent

class WPHCAgent(PHCAgent):
    def __init__(self, alpha = 0.5, gamma = 0.75, delta = 0.5, deltaLose = 1, numTraining = 0, **kwargs):
        super().__init__(alpha=alpha, gamma=gamma, delta=delta, numTraining=numTraining)
        self.deltaLose = deltaLose
        self.C = dict()
        self.PiAvg = dict()

    def accessPiAndInitAll(self, state, numActions):
        if state in self.Pi :
            return self.Pi[state]
        else:
            Q = self.Q[state] = np.full(numActions, 0.)
            Pi = self.Pi[state] = np.full(numActions, 1. / numActions)
            self.C[state] = 0
            self.PiAvg[state] = np.full(numActions, 1. / numActions)
            return Pi
    
    def learn(self, state, gameState):
        prevState = self.previousState.state
        prevAction = self.previousState.action
        prevNumActions = self.previousState.numActions
        self.C[prevState] += 1
        Q = self.Q[prevState]
        Pi = self.Pi[prevState]
        C = self.C[prevState]
        PiAvg = self.PiAvg[prevState]
        PiAvg = PiAvg + (Pi - PiAvg) / C
        self.PiAvg[prevState] = PiAvg
        reward = self.rewards(gameState)
        if state in self.Q:
            reward += self.gamma * np.amax(self.Q[state])
        Q[prevAction] = (1 - self.alpha) * Q[prevAction] + self.alpha * reward
        PiQ = np.dot(Pi, Q)
        PiAvgQ = np.dot(PiAvg, Q)
        delta = self.delta if PiQ > PiAvgQ else self.deltaLose
        Pi[prevAction] += delta if  prevAction == np.argmax(Q) else -delta / (prevNumActions - 1)
        Pi = np.maximum(Pi, np.full(prevNumActions, 0.))
        self.Pi[prevState] = Pi / np.sum(Pi)
