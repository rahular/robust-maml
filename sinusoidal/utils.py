import hashlib
import os
import pickle
import random

import numpy as np
import scipy.stats as st
import torch

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", font="Times New Roman")


def set_seed(seed, cudnn=True):
    """
    Seed everything we can!
    Note that gym environments might need additional seeding (env.seed(seed)),
    and num_workers needs to be set to 1.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # note: the below slows down the code but makes it reproducible
    if (seed is not None) and cudnn:
        torch.backends.cudnn.deterministic = True


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


def get_path_from_args(args):
    """ Returns a unique hash for an argparse object. """
    args_str = str(args)
    path = hashlib.md5(args_str.encode()).hexdigest()
    return path


def get_base_path():
    p = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(p):
        return p
    raise RuntimeError(
        "I dont know where I am; please specify a path for saving results."
    )


def get_stats(losses):
    loss_mean = np.mean(losses)
    loss_std = st.sem(losses)
    loss_conf = np.mean(
        np.abs(
            st.t.interval(0.95, losses.size - 1, loc=loss_mean, scale=loss_std)
            - loss_mean
        )
    )
    return loss_mean, loss_conf


def plot_df(df, path):
    df.dropna()
    plot = sns.lineplot(data=df, x="grad_steps", y="loss", hue="k_shot", style="k_shot")
    sns.despine()
    plot.set(xlabel="Gradient Steps", ylabel="Mean Squared Error")
    plt.legend(title="K")
    plot.figure.savefig(
        os.path.join(path, "plot.pdf"), bbox_inches="tight", pad_inches=0
    )
