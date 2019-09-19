from joblib import dump, load

import numpy as np
from tqdm import tqdm

SAVED_MODEL_PATH = r'trained_models/one_step_sarsa.joblib'


def one_step_sarsa_learn(
        env,
        n_episodes,
        alpha,
        gamma,
        epsilon,
        eps_decay,
        fa_type='linear',
        verbose=False,
        render=False,
        save_model=False,
        **kwargs
):
    # other func approx types are currently not supported
    assert fa_type == 'linear'

    # x, y, vx, vy, angle, vangle, left_leg, right_leg = initial_state

    rewards = []

    # initializes weights
    n = env.observation_space.shape[0]
    m = env.action_space.n
    weight_size = n + m
    w = np.zeros(shape=(weight_size,))

    # initializes epsilon schedule
    e_schedule = [epsilon * np.power(eps_decay, i) for i in range(n_episodes)]

    for i in tqdm(range(n_episodes)):
        # inits total reward
        total_reward = 0

        # resets env
        curr_state = env.reset()

        curr_e = e_schedule[i]

        curr_action = np.random.randint(0, m) \
            if np.random.rand() < curr_e \
            else _get_maxq_action(curr_state, w, m)

        done = False

        while not done:

            if render:
                env.render()

            # make a step
            next_state, r, done, info = env.step(curr_action)
            total_reward += r

            # # update weights if last step
            # if done:
            #     grad = _get_fa_gradient(curr_state, curr_action, m)
            #     td_target = r
            #     td_error = td_target - _q_value(curr_state, curr_action, w, m)
            #     delta_w = alpha * td_error * grad
            #     w += delta_w
            # else:

            # choose next action
            next_action = np.random.randint(0, m) \
                if np.random.rand() < curr_e \
                else _get_maxq_action(next_state, w, m)

            # update weights if regular step
            grad = _get_fa_gradient(curr_state, curr_action, m)
            td_target = r if done else r + gamma * _q_value(next_state, next_action, w, m)
            td_error = td_target - _q_value(curr_state, curr_action, w, m)
            delta_w = alpha * td_error * grad
            w += delta_w

            # update state & action
            curr_state = next_state
            curr_action = next_action

        if verbose:
            print(f'Episode: {i}.')
            print(f'Total Reward: {total_reward}')

        rewards.append(total_reward)

    if save_model:
        dump(w, SAVED_MODEL_PATH, compress=1)

    return rewards


def one_step_sarsa_predict(
        env,
        n_episodes,
        render=False,
        **kwargs
):
    rewards = []

    for i in tqdm(range(n_episodes)):
        # inits total reward
        total_reward = 0

        # resets env
        curr_state = env.reset()

        # loads weights
        w = load(SAVED_MODEL_PATH)

        m = env.action_space.n
        curr_action = _get_maxq_action(curr_state, w, m)

        done = False

        while not done:

            if render:
                env.render()

            # make a step
            next_state, r, done, info = env.step(curr_action)
            total_reward += r

            # choose next action
            curr_action = _get_maxq_action(next_state, w, m)

        rewards.append(total_reward)

    return rewards


def _q_value(state, action, w, m):
    s_a = _get_fa_gradient(state, action, m)
    q = np.dot(w, s_a)
    return q


def _get_maxq_action(state, w, m):
    qs = np.array([_q_value(state, action, w, m) for action in range(m)])
    return np.argmax(qs)


def _get_fa_gradient(state, action, m):
    s_comp = np.array(state, dtype=np.float)
    a_comp = np.zeros(shape=(m,))
    a_comp[action] = 1.0
    s_a = np.concatenate([s_comp, a_comp])
    return s_a
