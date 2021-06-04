#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from imp import init_builtin
from operator import le
import numpy as np
from tensorflow.python.keras.backend import update
from game import Agent
from agentUtil import *
import tensorflow as tf


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
        C=4,
        learningRate=0.005,
        **kwargs,
    ):
        self.C = int(C)
        self.learningRate = float(learningRate)
        self.it = 0
        self.QNetwork = None
        self.QQNetwork = None
        self.fitHistory = None
        self.architecture = (10, 5)
        self.convolutionalArchitecture = [(16, 2, 4), (8, 2, 2)]

    def __str__(self) -> str:
        return str("{:0.6f}".format(self.fitHistory.history["loss"][0]))

    def __call__(self, x):
        return self.QNetwork(x)

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
        return self.QNetwork(x)

    def inferQQ(self, x):
        return self.QQNetwork(x)


class DQNState:
    def __init__(self, state, action):
        self.state = state
        self.action = action
        self.outcomeState = None
        self.outcomeReward = None
        self.outcomeAction = None
        self.isTerminalOutcome = None

    def setOutcome(self, state, actions, gameState, rewards):
        self.outcomeState = state
        self.outcomeActions = actions
        self.outcomeReward = rewards(gameState)
        self.isTerminalOutcome = gameState.isWin() or gameState.isLose()


class DQNAgent(Agent):
    def __init__(
        self,
        K=4,
        gamma=0.99,
        epsilon=1.0,
        minibatchSize=32,
        experienceSize=500,
        finalEpsilon=0.1,
        numTraining=0,
        printSteps=5,
        **kwargs,
    ):
        self.index = 0
        self.metrics = {"meanGameScore": 0, "maxScore": 0}
        self.actionIt = 0
        self.episodeIt = 0
        self.experienceIt = 0
        self.previousState = None
        self.K = int(K)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.minibatchSize = int(minibatchSize)
        self.finalEpsilon = float(finalEpsilon)
        self.printSteps = int(printSteps)
        self.numTraining = int(numTraining)
        self.experienceSize = int(experienceSize)
        self.experienceReplay = [None] * int(experienceSize)
        self.rewards = Rewards(**kwargs)
        self.network = DQNNetwork(**kwargs)
        self.random = np.random.default_rng(int(kwargs["seed"])) if "seed" in kwargs else np.random.default_rng(12345)
        self.epsilonStep = (self.epsilon - self.finalEpsilon) / (self.numTraining if self.numTraining > 0 else 1)
        self.epsilonArray = [self.epsilon, 1 - self.epsilon]

    def registerInitialState(self, gameState):
        self.rewards.initial(gameState)
        if self.episodeIt == 0:
            print("Creating net")
            stateShape = gameStateTensor(gameState).shape
            stateShape[2] = self.K
            self.network.initNetworks(stateShape)
        self.actionIt = 0

    def selectMiniBatch(self):
        size = self.experienceIt if self.experienceIt < self.experienceSize else self.experienceSize
        minibatchSize = min(size, self.minibatchSize)
        miniBatchIndexes = self.random.choice(size, size=minibatchSize, replace=False)
        return miniBatchIndexes

    def updateExperience(self, state):
        self.experienceReplay[self.experienceIt % self.experienceSize] = state
        self.experienceIt += 1

    def final(self, gameState):
        state = gameStateTensor(gameState)
        if self.previousState != None:
            self.updateExperience(state, [0], gameState)
            self.learn()
            self.QQNetwork = tf.keras.models.clone_model(self.QNetwork)
        
        self.previousState = None
        if self.episodeIt < self.numTraining:
            self.epsilon -= self.epsilonStep
        self.episodeIt += 1
        

    def learn(self):
        miniBatchIndexes = self.selectMiniBatch()
        miniBatchQ = np.array([self.experienceReplay[k].state for k in miniBatchIndexes])
        miniBatchQQ = np.array([self.experienceReplay[k].outcomeState for k in miniBatchIndexes])
        actions = np.array([self.experienceReplay[k].action for k in miniBatchIndexes])
        rewards = np.array([self.experienceReplay[k].outcomeReward for k in miniBatchIndexes])
        isTerminal = np.array([self.experienceReplay[k].isOutcomeTerminal for k in miniBatchIndexes])
        QQ = self.QQNetwork(miniBatchQQ).numpy()
        QQ = np.array([np.max(QQ[i, self.experienceReplay[k].outcomeActions]) for i, k in enumerate(miniBatchIndexes)])
        y = self.gamma * QQ
        y[isTerminal == True] = 0
        y += rewards
        Q = self.QNetwork(miniBatchQ).numpy()
        Q[tuple(range(actions.size)), tuple(actions)] = y
        self.meanLoss += self.QNetwork.fit(miniBatchQ, Q, verbose=0, steps_per_epoch=len(miniBatchIndexes)).history[
            "loss"
        ][0]
        if (self.step % self.C) == 0:
            self.QQNetwork = tf.keras.models.clone_model(self.QNetwork)
        self.step += 1

    def getActionDQN(self, gameState):
        state = gameStateTensor(gameState)
        actions, actionsIndexes = getActions(gameState)
        if self.previousState != None:
            self.updateExperience(state, actionsIndexes, gameState)
            self.learn()
        randomAction = self.random.choice([True, False], p=self.epsilonArray)
        if randomAction and (self.it < self.numTraining):
            actionIndex = self.random.integers(0, len(actions))
        else:
            Q = self.QNetwork(np.array([state])).numpy()[0]
            actionIndex = np.argmax(Q[actionsIndexes])
        self.previousState = DQNState(state, actionsIndexes[actionIndex])
        return actions[actionIndex]

    def getAction(self, gameState):
        if (self.actionIt % self.K) == 0:
            pass
        actions, actionsIndexes = getActions(gameState)
        self.actionIt += 1
        return actions[actionIndex]
