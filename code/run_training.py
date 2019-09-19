import random

import numpy as np
import gym

from modeling.random_action import random_action
from modeling.one_step_sarsa import one_step_sarsa_learn
from modeling.one_step_sarsa_discrete import one_step_sarsa_discr_learn
from modeling.dqn import dqn_learn

MODEL = 0  # 0 - random; 1 - one-step SARSA; 2 - discrete-state SARSA; 3 - DQN

VERBOSE = False
RENDER = False
SAVE_MODEL = True

MODEL_PARAMS = {
    0: {
        'n_episodes': 1000,
    },
    1: {
        'n_episodes': 20000,
        'gamma': 0.9,
        'alpha': 0.1,
        'epsilon': 1.0,
        'eps_decay': 0.98,
        'fa_type': 'linear',
    },
    2: {
        'n_episodes': 20000,
        'gamma': 0.95,
        'alpha': 0.2,
        'epsilon': 1.0,
        'eps_decay': 0.98,
    },
    3: {
        'n_episodes': 1e6,
        'alpha': 1e-4,
    }
}

SEED = 1


def learn_to_fly():
    # inits env
    print('Initializing env.')
    env = gym.make('LunarLander-v2')

    # predefines seed
    env.seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # runs learning algorithm
    learn_func = _get_learn_function()

    print('Running learning.')
    n_episodes = MODEL_PARAMS[MODEL]['n_episodes']

    rewards = learn_func(
        env,

        # logging
        verbose=VERBOSE,
        render=RENDER,
        save_model=SAVE_MODEL,

        # model params
        **MODEL_PARAMS[MODEL],
    )

    print(f'\nTraining finished on total of {n_episodes} episodes.')
    print(f'Average Reward is {np.mean(np.array(rewards))}.')
    print(f'Last MA-100 Reward is {np.mean(np.array(rewards[-100:]))}.')

    env.close()


def _get_learn_function():
    if MODEL == 0:
        print('Random Action model selected.')
        return random_action
    elif MODEL == 1:
        print('1-Step SARSA model selected.')
        return one_step_sarsa_learn
    elif MODEL == 2:
        print('1-Step Discrete-State SARSA model selected.')
        return one_step_sarsa_discr_learn
    elif MODEL == 3:
        print('DQN model selected.')
        return dqn_learn
    else:
        raise ValueError('Invalid Model ID.')


if __name__ == '__main__':
    learn_to_fly()
