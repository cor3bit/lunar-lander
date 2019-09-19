import random

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from modeling.random_action import random_action
from modeling.one_step_sarsa import one_step_sarsa_predict
from modeling.one_step_sarsa_discrete import one_step_sarsa_discr_predict

N_EPISODES = 100

MODEL = 0  # 0 - random; 1 - one-step SARSA; 2 - discrete-state SARSA

RENDER = True
PLOT = True

SEED = 1


def predict():
    # inits env
    print('Initializing env.')
    env = gym.make('LunarLander-v2')

    # predefines seed
    env.seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # runs learning algorithm
    predict_func = _get_predict_function()

    print('Running prediction.')

    rewards = predict_func(
        env,

        # run params
        n_episodes=N_EPISODES,

        # logging
        render=RENDER,
    )

    print(f'\nFull Avg Reward is {np.mean(np.array(rewards))}.')
    print(f'Last-100 Avg Reward is {np.mean(np.array(rewards[-100:]))}.')

    env.close()

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

    print('Finished.')


def _get_predict_function():
    if MODEL == 0:
        print('Random Action model selected.')
        return random_action
    elif MODEL == 1:
        print('1-Step SARSA model selected.')
        return one_step_sarsa_predict
    elif MODEL == 2:
        print('1-Step Discrete-State SARSA model selected.')
        return one_step_sarsa_discr_predict
    else:
        raise ValueError('Invalid Model ID.')


if __name__ == '__main__':
    predict()
