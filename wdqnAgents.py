#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from agent import PacmanState
from agentUtil import *
from ast import literal_eval
from collections import deque
from dqnAgents import DQNAgent
from wphcAgents import WPHCAgent


class DQNTransition:
    def __init__(self, state, action, nextState, reward, isTerminal, stateW=None):
        self.state = state
        self.action = action
        self.nextState = nextState
        self.reward = reward
        self.isTerminal = isTerminal
        self.stateW = stateW


class DQNHistory:
    def __init__(self, K):
        self.K = K
        self.stack = None
        self.size = self.K + 1

    def init(self, agentState):
        self.stack = deque([agentState for k in range(self.size)])

    def update(self, state):
        if len(self.stack) >= self.size:
            self.stack.popleft()
        self.stack.append(state)

    def phi(self):
        if self.K == 1:
            return self.stack[1].state[0]
        stack = list(self.stack)[-self.K :]
        return np.array([state.state[0] for state in stack])

    def phiPrev(self):
        if self.K == 1:
            return self.stack[0].state[0]
        stack = list(self.stack)[: self.K]
        return np.array([state.state[0] for state in stack])

    def reward(self):
        return self.stack[-1].reward

    def action(self):
        return self.stack[-1].action

    def getTransition(self):
        return DQNTransition(
            self.phiPrev(),
            self.action(),
            self.phi(),
            self.reward(),
            self.stack[-1].isTerminal,
            self.stack[self.K - 1].state[1],
        )


class WDQNAgent(DQNAgent):
    def __init__(
        self,
        K=1,
        gamma=0.95,
        numEpochs=100,
        **kwargs,
    ):
        super().__init__(K=int(K), **kwargs, recurrentNetwork=int(K) > 1, gamma=float(gamma), sameActionPolicy=0)
        self.wphcAgent = WPHCAgent(gamma=float(gamma), **kwargs)
        self.sameActionPolicy = 0
        self.gameHistory = DQNHistory(self.K)
        self.whpcActions = 0
        self.numEpochs = int(numEpochs)
        self.parameters["numEpochs"] = self.numEpochs
        self.updateEpsilon = True

    def getState(self, gameState):
        matrix = gameStateTensor(gameState)
        matrix = (matrix - np.mean(matrix)) / np.std(matrix)
        wphcVector = self.wphcAgent.getState(gameState) if self.episodeIt < self.numExplore else None
        return tuple([matrix, wphcVector])

    def agentInit(self, gameState):
        if self.fromSaved:
            self.network.load(self.Qname, self.QQname)
            return
        state = self.getState(gameState)[0]
        stateShape = state.shape
        if self.recurrentNetwork:
            self.network.initNetworks(tuple([self.K]) + stateShape)
        else:
            self.network.initNetworks((stateShape[0], stateShape[1], self.K))
        self.parameters.update(self.network.toJson())

    def initState(self, agentState):
        agentState.validActionsIndexes = list(range(self.numActions))
        if self.previousState == None:
            self.gameHistory.init(agentState)
        else:
            self.gameHistory.update(agentState)
        if self.episodeIt < self.numTraining and self.previousState != None:
            self.updateExperience(self.gameHistory.getTransition())
        if self.episodeIt < self.numExplore:
            self.wphcAgent.initValues(agentState.state[1], self.numActions)
            if self.previousState != None:
                self.wphcAgent.learnStep(
                    self.previousState.state[1],
                    self.previousState.action,
                    self.numActions,
                    agentState.state[1],
                    agentState.reward,
                )
        if self.episodeIt == self.numExplore and self.actionIt == 0:
            self.whpcActions = self.totalActionIt
            self.trainNetwork()
            self.wphcAgent.updateJson()
            self.parameters["whpcTotalActions"] = self.whpcActions
            self.parameters = self.wphcAgent.parameters | self.parameters

    def trainNetwork(self):
        size = min(self.experienceIt, self.experienceSize)
        x = np.array([self.experienceReplay[k].state for k in range(size)])
        y = np.array([self.wphcAgent.Q[self.experienceReplay[k].stateW] for k in range(size)])
        self.network.learnQ(x=x, y=y, verbose=1, epochs=self.numEpochs)
        self.network.updateNetwork(0)

    def learn(self, agentState):
        if self.episodeIt < self.numTraining and self.episodeIt > self.numExplore:
            actionIt = self.totalActionIt - self.whpcActions + 1
            if (actionIt % self.trainUpdates) == 0:
                self.trainStep()
            self.network.updateNetwork(actionIt)

    def selectAction(self, agentState):
        if self.episodeIt < self.numExplore:
            return DIRECTIONS[self.wphcAgent.selectActionNum(agentState.state[1])]
        Q = self.network(np.array([self.gameHistory.phi()])).flatten()
        maxQIndex = np.argwhere(Q == np.amax(Q)).flatten()
        maxQIndex = maxQIndex[0] if maxQIndex.size == 1 else self.random.choice(maxQIndex)
        return DIRECTIONS[maxQIndex]
