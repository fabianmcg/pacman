#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from game import Agent
from agentUtil import *

class PHCAgent(Agent):
    def __init__(self, alpha = 0.25, gamma = 0.75, delta = 0.75, numTraining = 0, **kwargs):
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
    
    def registerInitialState(self, gameState):
        self.rewards.initial(gameState)
    
    def final(self, state):
        self.learn(self.previousState.state, state)
        self.previousState = None
        self.rewards.reset()
        self.it += 1

    def accessPiAndInitAll(self, state, numActions):
        if state in self.Pi:
            return self.Pi[state]
        else:
            Q = self.Q[state] = np.full(numActions, 0.)
            Pi = self.Pi[state] = np.full(numActions, 1. / numActions)
            return Pi
    
    def learn(self, state, gameState):
        prevState = self.previousState.state
        prevAction = self.previousState.action
        prevNumActions = self.previousState.numActions
        Q = self.Q[prevState]
        Pi = self.Pi[prevState]
        reward = self.rewards(gameState)
        if state in self.Q:
            reward += self.gamma * np.amax(self.Q[state])
        Q[prevAction] = (1 - self.alpha) * Q[prevAction] + self.alpha * reward
        Pi[prevAction] += self.delta if  prevAction == np.argmax(Q) else -self.delta / (prevNumActions - 1)
        Pi = np.maximum(Pi, np.full(prevNumActions, 0.))
        self.Pi[prevState] = Pi / np.sum(Pi)

    def getAction(self, gameState):
        actions = gameState.getLegalActions()
        serializedState = tuple(gameStateVector(gameState))
        if self.previousState != None:
            self.learn(serializedState, gameState)
        Pi = self.accessPiAndInitAll(serializedState, len(actions))
        action = self.random.choice(actions, p=Pi)
        self.previousState = QState(serializedState, actions.index(action), len(actions))
        return action
 
