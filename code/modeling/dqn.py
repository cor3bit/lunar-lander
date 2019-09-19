import random

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

SAVED_MODEL_PATH = r'trained_models/dqn_weights.h5f'
SAVED_REW_PATH = r'trained_models/dqn_rewards.joblib'


def dqn_learn(
        env,
        n_episodes,
        alpha,
        verbose=False,
        render=False,
        save_model=False,
        **kwargs
):
    n_actions = env.action_space.n

    # defines NN architecture
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(n_actions))
    model.add(Activation('linear'))

    memory = SequentialMemory(limit=1000000, window_length=1)

    policy = EpsGreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=n_actions,
        policy=policy,
        memory=memory,
        nb_steps_warmup=30000,
        target_model_update=1e-2
    )

    dqn.compile(Adam(lr=alpha), metrics=['mae'])

    dqn.fit(env, nb_steps=n_episodes, log_interval=1e5)

    dqn.save_weights(SAVED_MODEL_PATH, overwrite=save_model)


def dqn_predict(
        env,
        n_episodes,
        render=False,
        **kwargs
):
    n_actions = env.action_space.n

    # model
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(n_actions))
    model.add(Activation('linear'))

    memory = SequentialMemory(limit=1000000, window_length=1)

    policy = EpsGreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=n_actions,
        policy=policy,
        memory=memory,
        nb_steps_warmup=30000,
        target_model_update=1e-2
    )

    dqn.compile(Adam(lr=1e-4), metrics=['mae'])

    dqn.load_weights(SAVED_MODEL_PATH)

    rew_history = dqn.test(env, nb_episodes=n_episodes, visualize=render)

    rewards = rew_history.history['episode_reward']

    return list(rewards)
