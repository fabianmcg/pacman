#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.python.eager.backprop_util import IsTrainable
from PacmanAgent import PacmanAgent, PacmanState
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
        model.add(Dense(layer, activation="relu", init="he_uniform"))
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
        self.architecture = tuple([256]) if arch == None else literal_eval(arch)
        self.architecture = self.architecture + tuple([numActions])
        self.convolutionalArchitecture = (
            [(64, 4, 3, "relu"), (64, 3, 2, "relu")] if convArch == None else literal_eval(convArch)
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
        alpha=0.2,
        gamma=0.99,
        minibatchSize=32,
        experienceSize=100000,
        subselectScheme=None,
        clipValues=None,
        noStop=None,
        recurrent=None,
        fullQ=None,
        trainEvery=1,
        initialGames=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.experienceIt = 0
        self.K = int(K)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.minibatchSize = int(minibatchSize)
        self.experienceSize = int(experienceSize)
        self.experienceReplay = [None] * int(experienceSize)
        self.noStop = True if noStop == None else literal_eval(noStop)
        self.recurrentNetwork = False if recurrent == None else literal_eval(recurrent)
        self.subselectScheme = False if subselectScheme == None else literal_eval(subselectScheme)
        self.clipValues = True if clipValues == None else literal_eval(clipValues)
        self.fullQ = False if fullQ == None else literal_eval(fullQ)
        self.trainEvery = int(trainEvery)
        self.numActions = 4 if self.noStop else 5
        self.initialGames = int(initialGames)
        self.network = DQNNetwork(numActions=self.numActions, recurrentNetwork=self.recurrentNetwork, **kwargs)
        self.gameHistory = None
        self.parameters.update(
            {
                "K": self.K,
                "gamma": self.gamma,
                "experienceSize": self.experienceSize,
                "minibatchSize": self.minibatchSize,
                "subselectScheme": self.subselectScheme,
                "clipValues": self.clipValues,
                "experience": 0,
                "noStop": self.noStop,
                "numAction": self.numActions,
            }
        )

    def updateJson(self):
        self.parameters["learningEpochs"] = self.network.it
        self.parameters["experience"] = self.experienceIt

    def agentInit(self, gameState):
        state = gameStateTensor(gameState)
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
        size = self.experienceIt if self.experienceIt < self.experienceSize else self.experienceSize
        minibatchSize = min(size, self.minibatchSize)
        miniBatchIndexes = self.random.choice(size, size=minibatchSize, replace=False)
        return miniBatchIndexes

    def learn(self):
        miniBatchIndexes = self.selectMiniBatch()
        x = np.array([self.experienceReplay[k].state for k in miniBatchIndexes])
        xx = np.array([self.experienceReplay[k].nextState for k in miniBatchIndexes])
        actions = np.array([self.experienceReplay[k].action for k in miniBatchIndexes])
        rewards = np.array([self.experienceReplay[k].reward for k in miniBatchIndexes])
        isTerminal = np.array([self.experienceReplay[k].isTerminal for k in miniBatchIndexes])
        y = self.network.inferQ(x)
        yy = self.network.inferQQ(xx)
        if self.subselectScheme:
            validActions = [self.experienceReplay[k].validActions for k in miniBatchIndexes]
            yy = self.gamma * np.array([np.max(yy[i, actions]) for i, actions in enumerate(validActions)])
        else:
            yy = self.gamma * np.max(yy, axis=1)
        yy[isTerminal == True] = 0.0
        yy += rewards
        if self.fullQ:
            yy = y[tuple(range(actions.size)), tuple(actions)] * (1 - self.alpha) + self.alpha * yy
        if self.clipValues:
            yy = np.clip(yy, -1, 1)
        y[tuple(range(actions.size)), tuple(actions)] = yy
        self.network.learn(x, y, len(miniBatchIndexes))

    def beginGame(self, gameState):
        state = gameStateTensor(gameState)
        self.gameHistory = DQNHistory(self.K * 2, self.K, self.recurrentNetwork, state)

    def endGame(self, gameState):
        if self.episodeIt < self.numTraining:
            state = gameStateTensor(gameState)
            reward = self.rewards(gameState)
            if (self.episodeIt + 1) == self.numTraining:
                self.updateModel(state, reward, True, [0], True)
            else:
                self.updateModel(state, reward, True)
        self.gameHistory = None

    def updateModel(self, state, reward, isTerminal, validActions=[0], force=False):
        previousState = self.previousState
        if previousState != None:
            phiState = self.gameHistory.phi()
            phiNextState = self.gameHistory.phiNext(state)
            transition = DQNTransition(phiState, previousState.action, phiNextState, reward, isTerminal, validActions)
            self.updateExperience(transition)
            if (self.initialGames <= self.episodeIt) and (force or ((self.network.it % self.trainEvery) == 0)):
                self.learn()

    def selectAction(self, gameState, actions, actionsIndexes):
        if self.noStop:
            actions.remove(Directions.STOP)
            actionsIndexes.remove(DIR2CODE[Directions.STOP])
        state = gameStateTensor(gameState)
        reward = self.rewards(gameState)
        if self.episodeIt < self.numTraining:
            self.updateModel(state, reward, False, actionsIndexes)
        self.gameHistory.update(state)
        randomAction = self.random.choice([True, False], p=self.epsilonArray)
        if randomAction and (self.epsilon > 0.0):
            action = actions[self.random.integers(0, len(actions))]
        else:
            Q = self.network(np.array([self.gameHistory.phi()]))[0]
            # action = (
            #     DIRECTIONS[actionsIndexes[np.argmax(Q[actionsIndexes])]]
            #     if self.subselectScheme
            #     else DIRECTIONS[np.argmax(Q)]
            # )
            action = DIRECTIONS[actionsIndexes[np.argmax(Q[actionsIndexes])]]
        self.previousState = PacmanState(state, DIR2CODE[action], reward, False)
        if action not in actions:
            action = Directions.STOP
        return action
