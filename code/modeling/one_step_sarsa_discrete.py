from joblib import dump, load

import numpy as np
from tqdm import tqdm

SAVED_MODEL_PATH = r'trained_models/one_step_sarsa_discr.joblib'
SAVED_REW_PATH = r'trained_models/one_step_sarsa_discr_rewards.joblib'

FEATURE_DIMS = [5, 5, 5, 5, 5, 5, 2, 2]

FEATURE_BOUNDS = [
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0],
]


def one_step_sarsa_discr_learn(
        env,
        n_episodes,
        alpha,
        gamma,
        epsilon,
        eps_decay,
        verbose=False,
        render=False,
        save_model=False,
        **kwargs
):
    # env params
    # n = env.observation_space.n
    m = env.action_space.n

    # start_e, end_e, e_steps, e_decay = epsilon
    # e_schedule = [epsilon * np.power(eps_decay, i) for i in range(n_episodes)]
    e_schedule = _get_epsilon_schedule(epsilon, eps_decay, n_episodes)

    # curr_e = start_e

    # inits Q-function
    Q = np.zeros(shape=(*FEATURE_DIMS, m), dtype=np.float)

    total_step_counter = 0

    # x, y, vx, vy, angle, vangle, left_leg, right_leg = initial_state

    rewards = []

    # initializes weights

    for i in tqdm(range(n_episodes), disable=verbose):
        # resets env
        curr_state = env.reset()
        curr_state_disc = _discretize_state(curr_state)

        # curr_e = get_epsilon(curr_e, epsilon, total_step_counter)
        curr_e = e_schedule[i]

        # inits total reward
        total_reward = 0

        curr_action = np.random.randint(0, m) \
            if np.random.rand() < curr_e \
            else np.argmax(Q[curr_state_disc])

        done = False
        j = 0

        while not done:

            if render:
                env.render()

            # make a step
            next_state, r, done, info = env.step(curr_action)
            next_state_discr = _discretize_state(next_state)
            total_reward += r

            # choose next action
            next_action = np.argmax(Q[next_state_discr])

            # update weights if regular step
            td_target = r if done else r + gamma * Q[next_state_discr][next_action]
            td_error = td_target - Q[curr_state_disc][curr_action]

            Q[curr_state_disc][curr_action] += alpha * td_error

            # update state & action
            curr_state_disc = next_state_discr
            curr_action = next_action

            j += 1
            total_step_counter += 1

        if verbose:
            print(f'Episode: {i}.')
            print(f'Epsilon: {curr_e}.')
            print(f'Steps: {j}.')
            print(f'Total Reward: {total_reward}')

        rewards.append(total_reward)

        if i % 100 == 0:
            print(f'Last-100 Avg Reward is {np.mean(np.array(rewards[-100:]))}.')

    if save_model:
        dump(Q, SAVED_MODEL_PATH, compress=1)

        rewards_for_chart = np.array([list(range(n_episodes)), rewards])
        dump(rewards_for_chart, SAVED_REW_PATH, compress=1)

    return rewards


def one_step_sarsa_discr_predict(
        env,
        n_episodes,
        render=False,
        **kwargs
):
    Q = load(SAVED_MODEL_PATH)

    rewards = []

    for i in tqdm(range(n_episodes)):
        # resets env
        curr_state = env.reset()
        curr_state_disc = _discretize_state(curr_state)

        # inits total reward
        total_reward = 0

        curr_action = np.argmax(Q[curr_state_disc])

        done = False
        while not done:

            if render:
                env.render()

            # make a step
            next_state, r, done, info = env.step(curr_action)
            next_state_discr = _discretize_state(next_state)

            total_reward += r

            # choose next action
            curr_action = np.argmax(Q[next_state_discr])

        rewards.append(total_reward)

    return rewards


def _discretize_state(state):
    return tuple(_discretize_state_feature(state[i], i) for i in range(len(state)))


def _discretize_state_feature(val, i):
    feat_min, feat_max = FEATURE_BOUNDS[i]
    feat_dims = FEATURE_DIMS[i]

    norm_val = (val + abs(feat_min)) / (feat_max - feat_min)
    discr_val = min(feat_dims - 1, max(0, int(round((feat_dims - 1) * norm_val))))

    return discr_val


def _get_epsilon_schedule(start_epsilon, eps_decay, n_episodes):
    # e_schedule = [epsilon * np.power(eps_decay, i) for i in range(n_episodes)]
    # e_schedule = np.linspace(start=epsilon, stop=0.001, num=n_episodes)

    a = np.linspace(start=start_epsilon, stop=0.5, num=int(0.1 * n_episodes))
    b = np.linspace(start=0.5, stop=0.1, num=int(0.5 * n_episodes))
    c = np.linspace(start=0.1, stop=0.0, num=int(0.4 * n_episodes))
    e_schedule = np.concatenate([a, b, c])

    return e_schedule
