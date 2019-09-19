from joblib import load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SAVED_REW_PATH = r'../trained_models/one_step_sarsa_discr_rewards.joblib'


def draw_charts():
    episode_rewards = load(SAVED_REW_PATH)

    ma_size = 100
    moving_avgs = np.convolve(episode_rewards[1, :], np.ones((ma_size)) / ma_size, mode='valid')

    score_th = 200
    score_line = np.full_like(moving_avgs, fill_value=score_th)

    df = pd.DataFrame(data=np.array([moving_avgs, score_line]).T,
                      columns=['MA Score', 'Threshold'])

    sns.lineplot(data=df)

    plt.xlabel('Episode')
    plt.ylabel('MA-100 Score')
    plt.show()


if __name__ == '__main__':
    sns.set()
    draw_charts()
