#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from game import Agent
from agentUtil import *

class PHCAgent(Agent):
    def __init__(self, alpha = 0.5, gamma = 0.75, delta = 0.5, numTraining = 0, **kwargs):
        self.it = 0
        self.index = 0
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.numTraining = int(numTraining)
        self.Q = dict()
        self.Pi = dict()
        self.random = np.random.default_rng(int(kwargs["seed"])) if "seed" in kwargs else np.random.default_rng(12345)
        self.rewards = Rewards(**kwargs)
        self.previousState = None
    
    def final(self, state):
        self.learn(self.previousState.state, state)
        self.previousState = None
        self.rewards.reset()
        self.it += 1

    def accessQPi(self, state, numActions):
        if state in self.Q :
            return self.Q[state], self.Pi[state]
        else:
            Q = self.Q[state] = np.full(numActions, 0.)
            Pi = self.Pi[state] = np.full(numActions, 1. / numActions)
            return Q, Pi
    
    def learn(self, state, gameState):
        Q = self.Q[self.previousState.state]
        Pi = self.Pi[self.previousState.state]
        reward = self.rewards(gameState)
        if state in self.Q:
            reward += self.gamma * np.amax(self.Q[state])
        Q[self.previousState.action] = (1 - self.alpha) * Q[self.previousState.action] + self.alpha * reward
        Pi[self.previousState.action] += self.delta if  self.previousState.action == np.argmax(Q) else -self.delta / (self.previousState.numActions - 1)
        Pi = np.maximum(Pi, np.full(self.previousState.numActions, 0.))
        self.Pi[self.previousState.state] = Pi / np.sum(Pi)

    def getAction(self, gameState):
        actions = gameState.getLegalActions()
        serializedState = tuple(gameStateVectorPacked(gameState))
        Q, Pi = self.accessQPi(serializedState, len(actions))
        action = self.random.choice(actions, p=Pi)
        if self.previousState != None:
            self.learn(serializedState, gameState)
        self.previousState = QState(serializedState, actions.index(action), len(actions))
        return action
 
