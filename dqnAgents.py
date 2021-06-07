#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.python.keras.engine import training
from agent import PacmanAgent, PacmanState
from agentUtil import *
import tensorflow as tf
from ast import literal_eval
from collections import deque


class ClippedMeanSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        error = tf.clip_by_value(y_pred - y_true, clip_value_min=-1, clip_value_max=1)
        return tf.reduce_mean(tf.square(error), axis=-1)


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
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
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
        model.add(Dense(layer, activation="relu"))
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
        clipLoss=False,
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
        self.lossFunction = ClippedMeanSquaredError() if clipLoss else tf.losses.MeanSquaredError()

    def __str__(self) -> str:
        return str("{:0.6f}".format(self.fitHistory.history["loss"][0]))

    def __call__(self, x):
        return self.QNetwork(x).numpy()

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
                self.convolutionalArchitecture, self.architecture, stateShape, optimizer, self.lossFunction
            )
        else:
            self.QNetwork = convolutionalNetwork(
                self.convolutionalArchitecture, self.architecture, stateShape, optimizer, self.lossFunction
            )
        self.QNetwork.summary()
        self.QQNetwork = tf.keras.models.clone_model(self.QNetwork)

    def updateNetwork(self, updateIt):
        if (updateIt % self.C) == 0:
            self.QQNetwork = tf.keras.models.clone_model(self.QNetwork)

    def learn(self, x, y, epochSize):
        self.fitHistory = self.QNetwork.fit(x, y, verbose=0, steps_per_epoch=epochSize)
        self.it += 1

    def inferQ(self, x):
        return self.QNetwork(x, training=False).numpy()

    def inferQQ(self, x):
        return self.QQNetwork(x, training=False).numpy()

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


class DQNTransition:
    def __init__(self, state, action, nextState, reward, isTerminal):
        self.state = state
        self.action = action
        self.nextState = nextState
        self.reward = reward
        self.isTerminal = isTerminal


class DQNHistory:
    def __init__(self, K, isDiscrete, recurrent):
        self.K = K
        self.isDiscrete = isDiscrete
        self.recurrentOrder = recurrent
        self.stack = None
        self.size = K * 2 if self.isDiscrete else self.K + 1

    def init(self, agentState):
        self.stack = deque([agentState for k in range(self.size)])

    def update(self, state):
        if len(self.stack) >= self.size:
            self.stack.popleft()
        self.stack.append(state)

    def phi(self):
        if self.K == 1:
            return self.stack[1].state
        stack = list(self.stack)[-self.K :]
        if self.recurrentOrder:
            return np.array([state.state for state in stack])
        return np.concatenate(tuple([state.state for state in stack]), axis=2)

    def phiPrev(self):
        if self.K == 1:
            return self.stack[0].state
        stack = list(self.stack)[: self.K]
        if self.recurrentOrder:
            return np.array([state.state for state in stack])
        return np.concatenate(tuple([state.state for state in stack]), axis=2)

    def reward(self):
        if self.isDiscrete:
            stack = list(self.stack)[-self.K :]
            return np.sum(np.array([state.reward for state in stack]))
        return self.stack[-1].reward

    def action(self):
        if self.isDiscrete:
            return self.stack[self.K].action
        return self.stack[-1].action

    def getTransition(self):
        return DQNTransition(self.phiPrev(), self.action(), self.phi(), self.reward(), self.stack[-1].isTerminal)


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
        self.sameActionPolicy = self.K if self.sameActionPolicy >= 1 else 1
        self.network = DQNNetwork(
            numActions=self.numActions, recurrentNetwork=self.recurrentNetwork, clipLoss=self.clipValues, **kwargs
        )
        self.gameHistory = DQNHistory(self.K, self.sameActionPolicy > 1, self.recurrentNetwork)
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
        self.parameters["sameActionPolicy"] = self.sameActionPolicy

    def updateJson(self):
        self.parameters["learningEpochs"] = self.network.it
        self.parameters["experience"] = self.experienceIt

    def getState(self, gameState):
        matrix = gameStateTensor(gameState)
        matrix = (matrix - np.mean(matrix)) / np.std(matrix)
        return matrix

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

    def trainStep(self):
        if self.experienceIt == 0:
            return None
        miniBatchIndexes = self.selectMiniBatch()
        x = np.array([self.experienceReplay[k].state for k in miniBatchIndexes])
        xx = np.array([self.experienceReplay[k].nextState for k in miniBatchIndexes])
        actions = np.array([self.experienceReplay[k].action for k in miniBatchIndexes])
        rewards = np.array([self.experienceReplay[k].reward for k in miniBatchIndexes])
        isTerminal = np.array([self.experienceReplay[k].isTerminal for k in miniBatchIndexes])
        y = self.network.inferQ(x)
        yy = np.max(self.network.inferQQ(xx), axis=1) * self.gamma
        yy[isTerminal == True] = 0.0
        if self.clipValues:
            rewards = np.clip(rewards, -1.0, 1.0)
        yy += rewards
        y[tuple(range(actions.size)), tuple(actions)] = yy
        self.network.learn(x, y, len(miniBatchIndexes))

    def initState(self, agentState):
        if self.previousState == None:
            self.gameHistory.init(agentState)
        else:
            self.gameHistory.update(agentState)
        if self.episodeIt < self.numTraining and self.previousState != None:
            if (self.sameActionPolicy <= 1) or ((self.actionIt % self.sameActionPolicy) == 0):
                self.updateExperience(self.gameHistory.getTransition())
            elif agentState.isTerminal:
                left = self.sameActionPolicy - (self.actionIt % self.sameActionPolicy)
                for k in range(left):
                    self.gameHistory.update(PacmanState(state=agentState.state, reward=0, isTerminal=True))
                self.updateExperience(self.gameHistory.getTransition())

    def learn(self, agentState, isTerminal):
        if (self.episodeIt < self.numTraining):
            if ((self.totalActionIt % self.trainUpdates) == 0):
                self.trainStep()
            self.network.updateNetwork(self.totalActionIt)

    def selectAction(self, agentState):
        Q = self.network(np.array([self.gameHistory.phi()])).flatten()
        maxQIndex = np.argwhere(Q == np.amax(Q)).flatten()
        maxQIndex = maxQIndex[0] if maxQIndex.size == 1 else self.random.choice(maxQIndex)
        return DIRECTIONS[maxQIndex]
