import random

import numpy as np
import gym
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

SEED = 1

RENDER = True

PLOT = True

SAVED_MODEL_PATH = r'trained_models/dqn_weights.h5f'

if __name__ == '__main__':
    # inits env
    print('Initializing env.')
    env = gym.make('LunarLander-v2')

    # predefines seed
    env.seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

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

    rew_history = dqn.test(env, nb_episodes=100, visualize=RENDER)

    rewards = rew_history.history['episode_reward']

    print(np.mean(rewards))

    if PLOT:
        sns.set()

        rews = np.array(rewards)

        score_th = 200
        score_line = np.full_like(rews, fill_value=score_th)

        df = pd.DataFrame(data=np.array([rews, score_line]).T,
                          columns=['Score', 'Threshold'])

        sns.lineplot(data=df, marker='o')

        plt.xlabel('Episode')
        plt.ylabel('Test Episode Score')
        plt.show()
