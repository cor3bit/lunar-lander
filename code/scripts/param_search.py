import time

import numpy as np

from hyperopt import hp, tpe
from hyperopt.fmin import fmin

import gym

from modeling.one_step_sarsa import one_step_sarsa_learn

N_ITER_SEARCH = 100


def objective(params):
    env = gym.make('LunarLander-v2')

    avg_reward = one_step_sarsa_learn(
        env,

        gamma=0.95,

        alpha=params['alpha'],
        epsilon=params['epsilon'],

        verbose=False,
        render=False,
    )

    print("Avg Value {:.3f}, params {}".format(avg_reward, params))

    env.close()

    return -avg_reward


def search_hyperparams():
    start_t = time.time()

    print('\n')
    print('=' * 50)

    space = {
        'alpha': hp.uniform('alpha', 0.01, 1.0),
        'epsilon': hp.uniform('epsilon', 0.01, 1.0),
    }

    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=N_ITER_SEARCH)

    run_t_min = (time.time() - start_t) / 60.0

    print(f'\nRunning time: {run_t_min:.2f} min.')
    print(f'\nBest sln: {best}.')

    print('=' * 50)
    print('\n')


# --------------- Run Script ---------------

if __name__ == '__main__':
    search_hyperparams()
