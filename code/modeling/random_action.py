import numpy as np
from scipy import stats
from tqdm import tqdm


def random_action(
        env,
        n_episodes,
        verbose=False,
        render=False,
        **kwargs
):
    episode_total_rewards = []

    # initializes weights
    m = env.action_space.n

    all_states = []

    for i in tqdm(range(n_episodes), disable=verbose):
        # inits total reward
        total_reward = 0

        # resets env
        curr_state = env.reset()
        all_states.append(curr_state)

        curr_action = np.random.randint(0, m)
        done = False

        while not done:

            if render:
                env.render()

            # make a step
            next_state, r, done, info = env.step(curr_action)
            total_reward += r
            all_states.append(next_state)

            next_action = np.random.randint(0, m)
            curr_action = next_action

        if verbose:
            print(f'Episode: {i}.')
            print(f'Total Reward: {total_reward}')

        episode_total_rewards.append(total_reward)

    # analyze min/max
    all_states_arr = np.array(all_states)
    for col_name, col_vals in zip(['x', 'y', 'vx', 'vy', 'theta', 'vtheta', 'lleg', 'rleg'], all_states_arr.T):
        print(f'Feature {col_name}.')
        print(stats.describe(col_vals))

    return episode_total_rewards
