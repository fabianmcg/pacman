#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.python.eager.backprop_util import IsTrainable
from agent import PacmanAgent
from agentUtil import *
import tensorflow as tf
from ast import literal_eval
from collections import deque


def convolutionalNetwork(convolutionLayers, denseLayers, stateShape, optimizer, loss=tf.losses.MeanSquaredError()):
    from tensorflow.keras.layers import Dense, Flatten, Conv2D

    model = tf.keras.Sequential(name="DQN")
    model.add(
        Conv2D(
            convolutionLayers[0][0],
            convolutionLayers[0][1],
            strides=convolutionLayers[0][2],
            activation=convolutionLayers[0][3],
            input_shape=stateShape,
        )
    )
    for layer in convolutionLayers[1:]:
        model.add(Conv2D(layer[0], layer[1], strides=layer[2], activation=layer[3]))
    model.add(Flatten())
    for layer in denseLayers[0:-1]:
        model.add(Dense(layer, activation="relu"))
    model.add(tf.keras.layers.Dense(denseLayers[-1], activation="linear"))
    model.compile(loss=loss, optimizer=optimizer, metrics=["mean_squared_error", "accuracy"])
    return model


def recurrentConvolutionalNetwork(
    convolutionLayers, denseLayers, stateShape, optimizer, loss=tf.losses.MeanSquaredError()
):
    from tensorflow.keras.layers import Dense, Flatten, Conv2D, ConvLSTM2D

    model = tf.keras.Sequential(name="DQN")
    model.add(
        ConvLSTM2D(
            convolutionLayers[0][0],
            convolutionLayers[0][1],
            strides=convolutionLayers[0][2],
            input_shape=stateShape,
        )
    )
    for layer in convolutionLayers[1:]:
        model.add(Conv2D(layer[0], layer[1], strides=layer[2], activation=layer[3]))
    model.add(Flatten())
    for layer in denseLayers[0:-1]:
        model.add(Dense(layer, activation="relu", init=tf.keras.initializers.HeUniform()))
    model.add(tf.keras.layers.Dense(denseLayers[-1], activation="linear"))
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    return model


class DQNNetwork:
    def __init__(
        self,
        numActions,
        recurrentNetwork,
        C=200,
        learningRate=0.00025,
        arch=None,
        convArch=None,
        optimizer="RMSProp",
        **kwargs,
    ):
        self.it = 0
        self.C = int(C)
        self.numActions = numActions
        self.learningRate = float(learningRate)
        self.recurrentNetwork = recurrentNetwork
        self.QNetwork = None
        self.QQNetwork = None
        self.fitHistory = None
        self.architecture = tuple([512]) if arch == None else literal_eval(arch)
        self.architecture = self.architecture + tuple([numActions])
        self.convolutionalArchitecture = (
            [(32, 3, 1, "relu"), (64, 2, 1, "relu")] if convArch == None else literal_eval(convArch)
        )
        self.optimizerName = optimizer
        self.inputShape = None

    def __str__(self) -> str:
        return str("{:0.6f}".format(self.fitHistory.history["loss"][0]))

    def __call__(self, x):
        return self.QNetwork.predict(x)

    def initNetworks(self, stateShape):
        tf.device("/gpu:0")
        from tensorflow.keras.optimizers import Adam, RMSprop

        self.inputShape = stateShape

        optimizer = (
            RMSprop(learning_rate=self.learningRate)
            if self.optimizerName == "RMSProp"
            else Adam(learning_rate=self.learningRate)
        )
        if self.recurrentNetwork:
            self.QNetwork = recurrentConvolutionalNetwork(
                self.convolutionalArchitecture,
                self.architecture,
                stateShape,
                optimizer,
            )
        else:
            self.QNetwork = convolutionalNetwork(
                self.convolutionalArchitecture,
                self.architecture,
                stateShape,
                optimizer,
            )
        self.QNetwork.summary()
        self.QQNetwork = tf.keras.models.clone_model(self.QNetwork)

    def updateNetworks(self, force=False):
        if (self.it % self.C) == 0 or force:
            self.QQNetwork = tf.keras.models.clone_model(self.QNetwork)

    def learn(self, x, y, epochSize):
        self.fitHistory = self.QNetwork.fit(x, y, verbose=0, steps_per_epoch=epochSize)
        self.it += 1
        self.updateNetworks()

    def inferQ(self, x):
        return self.QNetwork.predict(x)

    def inferQQ(self, x):
        return self.QQNetwork.predict(x)

    def toJson(self):
        import json

        return {
            "architecture": self.architecture,
            "convolutionalArchitecture": self.convolutionalArchitecture,
            "learningRate": self.learningRate,
            "C": self.C,
            "inputShape": self.inputShape,
            "optimizerName": self.optimizerName,
            "recurrentNetwork": self.recurrentNetwork,
            "network": json.loads(self.QNetwork.to_json()),
        }


class DQNHistory:
    def __init__(self, size, K, recurrent, state):
        self.size = size
        self.K = K
        self.stack = deque([state for k in range(size)])
        self.recurrentOrder = recurrent

    def update(self, state):
        self.stack.popleft()
        self.stack.append(state)

    def phi(self):
        if (self.K - 1) == 0:
            return self.stack[-1]
        stack = list(self.stack)
        if self.recurrentOrder:
            return np.array([state for state in stack[-self.K :]])
        return np.concatenate(tuple([state for state in stack[-self.K :]]), axis=2)

    def phiNext(self, nextState):
        if (self.K - 1) == 0:
            return nextState
        stack = list(self.stack)
        sequence = stack[-(self.K - 1) :]
        sequence.append(nextState)
        if self.recurrentOrder:
            return np.array([state for state in sequence])
        return np.concatenate(tuple([state for state in sequence]), axis=2)


class DQNTransition:
    def __init__(self, state, action, nextState, reward, isTerminal, validActions):
        self.state = state
        self.action = action
        self.nextState = nextState
        self.reward = reward
        self.isTerminal = isTerminal
        self.validActions = validActions


class DQNAgent(PacmanAgent):
    def __init__(
        self,
        K=4,
        gamma=0.99,
        minibatchSize=32,
        experienceSize=200000,
        clipValues=None,
        recurrentNetwork=None,
        trainUpdates=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.experienceIt = 0
        self.K = int(K)
        self.gamma = float(gamma)
        self.minibatchSize = int(minibatchSize)
        self.experienceSize = int(experienceSize)
        self.experienceReplay = [None] * int(experienceSize)
        self.recurrentNetwork = False if recurrentNetwork == None else literal_eval(recurrentNetwork)
        self.clipValues = True if clipValues == None else literal_eval(clipValues)
        self.trainUpdates = int(trainUpdates)
        self.numActions = 4 if self.noStopAction else 5
        self.network = DQNNetwork(numActions=self.numActions, recurrentNetwork=self.recurrentNetwork, **kwargs)
        self.gameHistory = None
        self.parameters.update(
            {
                "K": self.K,
                "gamma": self.gamma,
                "experienceSize": self.experienceSize,
                "minibatchSize": self.minibatchSize,
                "clipValues": self.clipValues,
                "experience": 0,
                "numAction": self.numActions,
            }
        )

    def updateJson(self):
        self.parameters["learningEpochs"] = self.network.it
        self.parameters["experience"] = self.experienceIt

    def getState(self, gameState):
        return gameStateTensor(gameState)

    def agentInit(self, gameState):
        state = self.getState(gameState)
        stateShape = state.shape
        if self.recurrentNetwork:
            self.network.initNetworks(tuple([self.K]) + stateShape)
        else:
            self.network.initNetworks((stateShape[0], stateShape[1], self.K))
        self.parameters.update(self.network.toJson())

    def updateExperience(self, state):
        self.experienceReplay[self.experienceIt % self.experienceSize] = state
        self.experienceIt += 1

    def selectMiniBatch(self):
        size = min(self.experienceIt, self.experienceSize)
        minibatchSize = min(size, self.minibatchSize)
        miniBatchIndexes = self.random.choice(size, size=minibatchSize, replace=False)
        return miniBatchIndexes

    def beginGame(self, gameState):
        state = self.getState(gameState)
        self.gameHistory = DQNHistory(self.K * 2, self.K, self.recurrentNetwork, state)

    def endGame(self, gameState):
        self.gameHistory = None

    def trainStep(self):
        miniBatchIndexes = self.selectMiniBatch()
        x = np.array([self.experienceReplay[k].state for k in miniBatchIndexes])
        xx = np.array([self.experienceReplay[k].nextState for k in miniBatchIndexes])
        actions = np.array([self.experienceReplay[k].action for k in miniBatchIndexes])
        rewards = np.array([self.experienceReplay[k].reward for k in miniBatchIndexes])
        isTerminal = np.array([self.experienceReplay[k].isTerminal for k in miniBatchIndexes])
        y = self.network.inferQ(x)
        yy = np.max(self.network.inferQQ(xx), axis=1) * self.gamma
        yy[isTerminal == True] = 0.0
        yy += rewards
        if self.clipValues:
            yy = np.clip(yy, -1.0, 1.0)
        y[tuple(range(actions.size)), tuple(actions)] = yy
        self.network.learn(x, y, len(miniBatchIndexes))

    def initState(self, agentState):
        previousState = self.previousState
        if self.episodeIt < self.numTraining and previousState != None:
            phiState = self.gameHistory.phi()
            phiNextState = self.gameHistory.phiNext(agentState.state)
            transition = DQNTransition(
                phiState,
                previousState.action,
                phiNextState,
                agentState.reward,
                agentState.isTerminal,
                agentState.validActionsIndexes,
            )
            self.updateExperience(transition)

    def learn(self, agentState, isTerminal):
        if (self.episodeIt < self.numTraining) and (
            ((self.network.it % self.trainUpdates) == 0) or (isTerminal and (self.episodeIt + 1) == self.numTraining)
        ):
            self.trainStep()

    def selectAction(self, agentState):
        self.gameHistory.update(agentState.state)
        Q = self.network(np.array([self.gameHistory.phi()]))[0]
        action = DIRECTIONS[np.argmax(Q)]
        return action
