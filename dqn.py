#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from imp import init_builtin
from operator import le
import numpy as np
from game import Agent
from agentUtil import *
import tensorflow as tf

from pacman import GameState

# def network(hiddenLayers, unitsLast, stateShape, init, optimizer, loss):
#     from tensorflow.keras.models import Model
#     from tensorflow.keras.layers import Concatenate, Dense, Input, Flatten
#     modelInput = Input(shape=stateShape, name="Input")
#     layer = Flatten(name="Flatten")(modelInput)
#     for i, k in enumerate(hiddenLayers):
#         layer = Dense(k, activation='relu', kernel_initializer=init, name='H' + str(i))(layer)
#     modelOutput = Concatenate(name="Output")([Dense(1, name=str(k))(layer) for k in range(unitsLast)])
#     model = Model(inputs=modelInput, outputs=modelOutput)
#     model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
#     return model

def network(layersUnits, stateShape, optimizer, init = None, loss = tf.losses.MeanSquaredError()):
    from tensorflow.keras.layers import Dense, Flatten
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=stateShape))
    for x in layersUnits[0:-1]:
        model.add(Dense(x, activation='relu', kernel_initializer=init))
    model.add(tf.keras.layers.Dense(layersUnits[-1], activation='linear', kernel_initializer=init))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

class DQNAgent(Agent):
    def __init__(self, alpha = 0.25, gamma = 0.75, experienceSize = 400, C = 3, numTraining = 0, epsilon = 0.1, learningRate = 0.1, minibatchSize = 100, **kwargs):
        self.it = 0
        self.git = 0
        self.index = 0
        self.experienceIt = 0
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.numTraining = int(numTraining)
        self.experienceSize = int(experienceSize)
        self.experienceReplay = [None] * int(experienceSize)
        self.C = int(C)
        self.epsilon = float(epsilon)
        self.epsilonArray = [float(self.epsilon), 1 - float(self.epsilon)]
        self.learningRate = float(learningRate)
        self.minibatchSize = int(minibatchSize)
        self.rewards = Rewards(**kwargs)
        self.QNetwork = None
        self.QQNetwork = None
        self.random = np.random.default_rng(int(kwargs["seed"])) if "seed" in kwargs else np.random.default_rng(12345)
        self.architecture = (8, 4, 5)
        self.previousState = None
    
    def initNetworks(self, stateShape):
        tf.device("/gpu:0")
        from tensorflow.keras.initializers import RandomUniform, zeros
        self.QNetwork = network(self.architecture, stateShape, tf.keras.optimizers.Adam(learning_rate=self.learningRate), RandomUniform(minval=-0.05, maxval=0.05, seed=None))
        self.QQNetwork = network(self.architecture, stateShape, tf.keras.optimizers.Adam(learning_rate=self.learningRate), zeros())

    def registerInitialState(self, gameState):
        if self.it == 0:
            state = gameStateTensor(gameState)
            self.initNetworks(state.shape)
        if self.it > self.numTraining:
            self.epsilon = 0.01
            self.epsilonArray = [float(self.epsilon), 1 - float(self.epsilon)]

    def final(self, gameState):
        self.learn(gameStateTensor(gameState), gameState, [4])
        self.rewards.reset()
        self.previousState = None
        self.it += 1
        if (self.it % 5) == 0:
            print(self.it)
    
    def learn(self, state, gameState, actionsIndexes):
        reward = self.rewards(gameState)
        self.experienceReplay[self.experienceIt] = (self.previousState.state, self.previousState.action, reward, state, gameState.isWin() or gameState.isLose(), actionsIndexes)
        self.experienceIt = (self.experienceIt + 1) % self.experienceSize
        size = self.git if self.git < self.experienceSize else self.experienceSize
        minibatchSize = min(size, self.minibatchSize)
        miniBatchIndexes = self.random.choice(size, size=minibatchSize, replace=False)
        miniBatchQ = np.array([self.experienceReplay[k][0] for k in miniBatchIndexes])
        miniBatchQQ = np.array([self.experienceReplay[k][3] for k in miniBatchIndexes])
        isTerminal = np.array([self.experienceReplay[k][4] for k in miniBatchIndexes])
        actions = np.array([self.experienceReplay[k][1] for k in miniBatchIndexes])
        rewards = np.array([self.experienceReplay[k][2] for k in miniBatchIndexes])
        QQ = self.QQNetwork(miniBatchQQ).numpy()
        QQ = np.array([np.max(QQ[i, self.experienceReplay[k][5]]) for i, k in enumerate(miniBatchIndexes)])
        y = self.gamma * QQ
        y[isTerminal > 0] = 0
        y += rewards
        Q = self.QNetwork(miniBatchQ).numpy()
        Q[tuple(range(actions.size)), tuple(actions)] = y
        self.QNetwork.fit(miniBatchQ, Q, verbose=0)
        if self.git % self.C == 0:
            self.QQNetwork = tf.keras.models.clone_model(self.QNetwork)

    def getAction(self, gameState):
        state = gameStateTensor(gameState)
        actions, actionsIndexes = getActions(gameState)
        if self.previousState != None:
            self.learn(state, gameState, actionsIndexes)
        randomAction = self.random.choice([False, True], p=self.epsilonArray)
        if randomAction:
            action = self.random.integers(0, len(actions))
        else:
            Q = self.QNetwork(np.array([state])).numpy()[0]
            action = np.argmax(Q[actionsIndexes])
        self.previousState = QState(state, actionsIndexes[action])
        self.git += 1
        return actions[action]
