#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PacmanAgent import PacmanAgent, PacmanState
from agentUtil import *
import tensorflow as tf
from ast import literal_eval as make_tuple

def convolutionalNetwork(
    convolutionLayers, denseLayers, stateShape, optimizer, init=None, outInit=None, loss=tf.losses.MeanSquaredError()
):
    from tensorflow.keras.layers import Dense, Flatten, Conv2D

    model = tf.keras.Sequential()
    model.add(
        Conv2D(
            convolutionLayers[0][0],
            convolutionLayers[0][1],
            strides=convolutionLayers[0][2],
            activation="relu",
            kernel_initializer=init,
            input_shape=stateShape,
        )
    )
    for layer in convolutionLayers[1:]:
        model.add(Conv2D(layer[0], layer[1], strides=layer[2], activation="relu", kernel_initializer=init))
    model.add(Flatten())
    for layer in denseLayers[0:-1]:
        model.add(Dense(layer, activation="relu", kernel_initializer=init))
    model.add(tf.keras.layers.Dense(denseLayers[-1], activation="linear", kernel_initializer=outInit))
    model.compile(loss=loss, optimizer=optimizer, metrics=["mean_squared_error", "accuracy"])
    return model


class DQNNetwork:
    def __init__(
        self,
        C=1000,
        learningRate=0.05,
        arch = None,
        convArch = None,
        **kwargs,
    ):
        self.C = int(C)
        self.learningRate = float(learningRate)
        self.it = 0
        self.QNetwork = None
        self.QQNetwork = None
        self.fitHistory = None
        self.architecture = (256, 5) if arch == None else make_tuple(arch)
        self.convolutionalArchitecture = [(64, 2, 4)] if convArch == None else make_tuple(convArch)

    def __str__(self) -> str:
        return str("{:0.6f}".format(self.fitHistory.history["loss"][0]))

    def __call__(self, x):
        return self.QNetwork(x).numpy()

    def initNetworks(self, stateShape):
        tf.device("/gpu:0")
        from tensorflow.keras.initializers import RandomUniform

        self.QNetwork = convolutionalNetwork(
            self.convolutionalArchitecture,
            self.architecture,
            stateShape,
            tf.keras.optimizers.Adam(learning_rate=self.learningRate),
            RandomUniform(minval=-0.05, maxval=0.05, seed=None),
            RandomUniform(minval=-0.0005, maxval=0.0005, seed=None),
        )
        self.QQNetwork = tf.keras.models.clone_model(self.QNetwork)

    def updateNetworks(self, force=False):
        if (self.it % self.C) == 0 or force:
            self.QQNetwork = tf.keras.models.clone_model(self.QNetwork)

    def learn(self, x, y, epochSize):
        self.fitHistory = self.QNetwork.fit(x, y, verbose=0, steps_per_epoch=epochSize)
        self.it += 1
        self.updateNetworks()

    def inferQ(self, x):
        return self.QNetwork(x).numpy()

    def inferQQ(self, x):
        return self.QQNetwork(x).numpy()


class DQNHistory:
    def __init__(self, size, K, state):
        self.size = size
        self.K = K
        self.stack = [state for k in range(size)]

    def update(self, state):
        self.stack.pop(0)
        self.stack.append(state)

    def phi(self):
        return np.concatenate(tuple([state for state in self.stack[-self.K :]]), axis=2)

    def phiNext(self, nextState):
        sequence = self.stack[-(self.K - 1) :]
        sequence.append(nextState)
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
        experienceSize=100000,
        subselectScheme = True,
        clipValues = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.experienceIt = 0
        self.K = int(K)
        self.gamma = float(gamma)
        self.minibatchSize = int(minibatchSize)
        self.experienceSize = int(experienceSize)
        self.experienceReplay = [None] * int(experienceSize)
        self.network = DQNNetwork(**kwargs)
        self.gameHistory = None
        self.subselectScheme = bool(subselectScheme)
        self.clipValues = bool(clipValues)

    def agentInit(self, gameState):
        state = gameStateTensor(gameState)
        stateShape = state.shape
        self.network.initNetworks((stateShape[0], stateShape[1], self.K))

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
        validActions = [self.experienceReplay[k].validActions for k in miniBatchIndexes]
        y = self.network(x)
        yy = self.network.inferQQ(xx)
        if self.subselectScheme:
            yy = self.gamma * np.array([np.max(yy[i, actions]) for i, actions in enumerate(validActions)])
        else:
            yy = self.gamma * np.max(yy, axis=1)
        yy[isTerminal == True] = 0
        yy += rewards
        if self.clipValues:
            yy = np.clip(yy, -1, 1)
        y[tuple(range(actions.size)), tuple(actions)] = yy
        self.network.learn(x, y, len(miniBatchIndexes))

    def beginGame(self, gameState):
        state = gameStateTensor(gameState)
        self.gameHistory = DQNHistory(self.K + 2, self.K, state)

    def endGame(self, gameState):
        if self.episodeIt < self.numTraining:
            state = gameStateTensor(gameState)
            reward = self.rewards(gameState)
            self.train(state, reward, True)
        self.gameHistory = None

    def train(self, state, reward, isTerminal, validActions=[0]):
        previousState = self.previousState
        if previousState != None:
            phiState = self.gameHistory.phi()
            phiNextState = self.gameHistory.phiNext(state)
            transition = DQNTransition(phiState, previousState.action, phiNextState, reward, isTerminal, validActions)
            self.updateExperience(transition)
            self.learn()

    def selectAction(self, gameState, actions, actionsIndexes):
        state = gameStateTensor(gameState)
        reward = self.rewards(gameState)
        if self.episodeIt < self.numTraining:
            self.train(state, reward, False, actionsIndexes)
        self.gameHistory.update(state)
        randomAction = self.random.choice([True, False], p=self.epsilonArray)
        if randomAction and (self.epsilon > 0.):
            action = actions[self.random.integers(0, len(actions))]
        else:
            Q = self.network(np.array([self.gameHistory.phi()]))[0]
            action = DIRECTIONS[actionsIndexes[np.argmax(Q[actionsIndexes])]] if self.subselectScheme else DIRECTIONS[np.argmax(Q)]
        self.previousState = PacmanState(state, DIR2CODE[action], reward, False)
        return action
