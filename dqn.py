#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from imp import init_builtin
from operator import le
import numpy as np
from game import Agent
from agentUtil import *
import tensorflow as tf

def convolutionalNetwork(convLayers, denseLayers, stateShape, optimizer, init = None, loss = tf.losses.MeanSquaredError()):
    from tensorflow.keras.layers import Dense, Flatten, Conv2D
    model = tf.keras.Sequential()
    model.add(Conv2D(convLayers[0][0], convLayers[0][1], activation='relu', kernel_initializer=init, input_shape=stateShape))
    for x in convLayers[1:]:
        model.add(Conv2D(x[0], x[1], activation='relu', kernel_initializer=init))
    model.add(Flatten())
    for x in denseLayers[0:-1]:
        model.add(Dense(x, activation='relu', kernel_initializer=init))
    model.add(tf.keras.layers.Dense(denseLayers[-1], activation='linear', kernel_initializer=init))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

def network(layersUnits, stateShape, optimizer, init = None, loss = tf.losses.MeanSquaredError()):
    from tensorflow.keras.layers import Dense, Flatten
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=stateShape))
    for x in layersUnits[0:-1]:
        model.add(Dense(x, activation='relu', kernel_initializer=init))
    model.add(tf.keras.layers.Dense(layersUnits[-1], activation='linear', kernel_initializer=init))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

class DQNState:
    def __init__(self, state, action):
        self.state = state
        self.action = action
        self.outcomeState = None
        self.outcomeReward = None
        self.outcomeActions = None
        self.isOutcomeTerminal = None
    
    def setOutcome(self, state, actions, gameState, rewards):
        self.outcomeState = state
        self.outcomeActions = actions
        self.isOutcomeTerminal =  gameState.isWin() or gameState.isLose()
        self.outcomeReward = rewards(gameState)

class DQNAgent(Agent):
    def __init__(self, alpha = 0.25, gamma = 0.75, epsilon = 1., experienceSize = 500, minibatchSize = 200, C = 20, numTraining = 0, learningRate = 0.25, **kwargs):
        self.it = 0
        self.eit = 0
        self.step = 1
        self.index = 0
        self.experienceIt = 0
        self.QNetwork = None
        self.QQNetwork = None
        self.previousState = None
        self.C = int(C)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.learningRate = float(learningRate)
        self.minibatchSize = int(minibatchSize)
        self.numTraining = int(numTraining)
        self.experienceSize = int(experienceSize)
        self.experienceReplay = [None] * int(experienceSize)
        self.rewards = Rewards(**kwargs)
        self.random = np.random.default_rng(int(kwargs["seed"])) if "seed" in kwargs else np.random.default_rng(12345)
        self.epsilonStep = (self.epsilon - 0.001) / (self.numTraining if self.numTraining > 0 else 1)
        self.architecture = (10, 5)
        self.convArchitecture = [(1, 2), (1, 2)]
    
    def initNetworks(self, stateShape):
        tf.device("/gpu:0")
        from tensorflow.keras.initializers import RandomUniform, zeros
        self.QNetwork = convolutionalNetwork(self.convArchitecture, self.architecture, stateShape, 
            tf.keras.optimizers.Adam(learning_rate=self.learningRate), RandomUniform(minval=-0.0005, maxval=0.0005, seed=None))
        self.QQNetwork = tf.keras.models.clone_model(self.QNetwork)

    def epsilonArray(self):
        return [float(self.epsilon), 1 - float(self.epsilon)]

    def updateExperience(self, state, actions, gameState):
        self.previousState.setOutcome(state, actions, gameState, self.rewards)
        self.experienceReplay[self.experienceIt] = self.previousState
        self.experienceIt = (self.experienceIt + 1) % self.experienceSize
        self.eit += 1

    def selectMiniBatch(self):
        size = self.eit if self.eit < self.experienceSize else self.experienceSize
        minibatchSize = min(size, self.minibatchSize)
        miniBatchIndexes = self.random.choice(size, size=minibatchSize, replace=False)
        return miniBatchIndexes

    def registerInitialState(self, gameState):
        self.rewards.initial(gameState)
        if self.it == 0:
            state = gameStateTensor(gameState)
            self.initNetworks(state.shape)
        self.step = 1

    def final(self, gameState):
        state = gameStateTensor(gameState)
        if self.previousState != None:
            self.updateExperience(state, [4], gameState)
            self.learn()
            self.QQNetwork = tf.keras.models.clone_model(self.QNetwork)
        if self.it < self.numTraining:
            self.epsilon -= self.epsilonStep
        self.rewards.reset()
        self.previousState = None
        self.it += 1
        if (self.it % 10) == 0:
            print(self.it, gameState.getScore())
    
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
        y[isTerminal > 0] = 0
        y += rewards
        Q = self.QNetwork(miniBatchQ).numpy()
        Q[tuple(range(actions.size)), tuple(actions)] = y
        self.QNetwork.fit(miniBatchQ, Q, verbose=0, steps_per_epoch=len(miniBatchIndexes))
        if (self.step % self.C) == 0:
            self.QQNetwork = tf.keras.models.clone_model(self.QNetwork)
        self.step += 1

    def getAction(self, gameState):
        state = gameStateTensor(gameState)
        actions, actionsIndexes = getActions(gameState)
        if self.previousState != None:
            self.updateExperience(state, actionsIndexes, gameState)
            self.learn()
        randomAction = self.random.choice([True, False], p=self.epsilonArray())
        if randomAction and self.it < self.numTraining:
            action = self.random.integers(0, len(actions))
        else:
            Q = self.QNetwork(np.array([state])).numpy()[0]
            action = np.argmax(Q[actionsIndexes])
        self.previousState = DQNState(state, actionsIndexes[action])
        return actions[action]
