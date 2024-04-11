import os
import sys
import copy
import pathlib
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from umap import UMAP

os.chdir(os.path.expanduser('~/ryandew_adaptive_preference/RA'))

from backend.model import gp_inf, gp_inf_fix_amp

results_analysis_dir = pathlib.Path("results_analysis")
results_dfs_dir = results_analysis_dir.joinpath("results_dfs")
reoptim_forward_looking_dir = results_analysis_dir.joinpath("reoptimize_backend_forward_looking")
train_df = pd.read_csv(results_dfs_dir.joinpath("train_df.csv"))
test_df = pd.read_csv(results_dfs_dir.joinpath("test_df.csv"))

test_df["fixed_test_ratings"] = pd.Series(
    [
        str(
            list(
                np.array(
                    list(
                        eval(test_df["random_test_predictions_ratings"].iloc[i]).values(),
                    )
                )[:, 1]
            )
        )
        for i in range(test_df.shape[0])
    ]
)
test_df["fixed_test_items"] = pd.Series(
    [
        str(
            list(
                eval(test_df["random_test_predictions_ratings"].iloc[i]).keys(),
            )
        )
        for i in range(test_df.shape[0])
    ]
)

passed_ac_ids = train_df[train_df["passed_redisplay"] == True].index
train_passed_ac = train_df.loc[passed_ac_ids]
test_passed_ac = test_df.loc[passed_ac_ids]

assert all(train_passed_ac.index == test_passed_ac.index)

DATA_FILE = pathlib.Path("backend").joinpath("data", "dresses10_urls.csv")
all_items = pd.read_csv(DATA_FILE)


def unit_constrain(z: np.ndarray, padding=1e-4):
    return padding + (1 - 2 * padding) * (z - z.min()) / (z.max() - z.min())


z_unconstr_all = np.array(all_items.loc[:, "E1":].values)
z_all = np.apply_along_axis(unit_constrain, 0, z_unconstr_all)

from abc import ABC, abstractmethod


class PrefOptimBase(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.status = "train"
        self.y = None

    @abstractmethod
    def compute_utility_raw(self, z):
        """
        Compute the utility of z without considering the scaling issue.
        """
        pass

    def compute_utility(self, z):
        if self.status == "train":
            print("computing training utility")
            return self.compute_utility_raw(z)

        elif self.status == "test":
            print("computing test utility")
            return self.compute_utility_inverse_normalized(z)

        else:
            raise ValueError(f"Unimplemented status: {self.status}")

    def compute_utility_inverse_normalized(self, z):
        """
        Assuming that the model's saved y have already been normalized, (so that compute_utility_raw is also normalized),
        compute the un-normalized prediction.
        """

        norm_u = self.compute_utility_raw(z)
        if self.y_std == 0:
            return norm_u
        else:
            return norm_u * self.y_std + self.y_mean

    def ends_training(self):
        print("Model training is completed, switched to test mode")
        self.status = "test"

        self.y_mean = self.y.mean()
        self.y_std = self.y.std()

        if self.y_std != 0:
            self.y = (self.y - self.y_mean) / self.y_std


class PrefOptimOG(PrefOptimBase):
    def __init__(self, z_shown, ratings, opt_hypers=True, map=True):
        super().__init__()

        self.version = 2.0
        self.x = z_shown
        self.y = ratings
        self.map = map

        initial_hypers = [0.1, 0.4, 0.8]
        optim_args = {"x": self.x, "y": self.y}
        if opt_hypers:
            if map:
                self.hypers = gp_inf.map_hypers(initial_hypers, optim_args)
            else:
                self.hypers = gp_inf.optim_hypers(initial_hypers, optim_args)
        else:
            self.hypers = initial_hypers

    def ucb(self, z, scaling_factor):
        gp_mean, gp_cov = gp_inf.compute_gp(new_x=z, x=self.x, y=self.y, hypers=self.hypers)
        sigma = 2 * np.sqrt(np.diag(gp_cov))
        return gp_mean + scaling_factor * sigma

    def info_gain(self, z):
        gp_cov = gp_inf.compute_gp(new_x=z, x=self.x, y=self.y, hypers=self.hypers)[1]
        return 2 * np.sqrt(np.diag(gp_cov))

    def acquisition_fn(self, z):
        return self.info_gain(z)

    def update_posterior(self, new_rating, new_z, update_hypers=False):
        self.x = np.vstack([self.x, new_z])
        self.y = np.concatenate((self.y, new_rating))
        if update_hypers:
            optim_args = {"x": self.x, "y": self.y}
            initial_hypers = [0.1, 0.4, 0.8]
            if self.map:
                self.hypers = gp_inf.map_hypers(initial_hypers, optim_args)
            else:
                self.hypers = gp_inf.optim_hypers(initial_hypers, optim_args)

    def next_item(self, available_z):
        return np.argsort(self.acquisition_fn(available_z))[::-1]

    def compute_utility_raw(self, z):
        return gp_inf.compute_gp(new_x=z, x=self.x, y=self.y, hypers=self.hypers, compute_cov=False)

    def ends_training(self):
        super().ends_training()  # self.y has been normalized here in the super call

        initial_hypers = [0.1, 0.8]
        optim_args = {"x": self.x, "y": self.y}
        if self.map:
            self.hypers = gp_inf_fix_amp.map_hypers(initial_hypers, optim_args)
        else:
            self.hypers = gp_inf_fix_amp.optim_hypers(initial_hypers, optim_args)


class PrefOptimAgg(PrefOptimOG):

    # Keep the `opt_hypers` params here for API consistency
    def __init__(self, z_shown, ratings, opt_hypers=True, map=True):

        # However, when initializing, we are not actually doing optimization
        super().__init__(z_shown, ratings, opt_hypers=False, map=map)

        # Instead, use the hyper values computed from the 5.0 results,
        # using 100 people who saw 50 refine items
        self.hypers = {
            "noise": 0.3582851696037144,
            "amp": 0.5714172545679208,
            "ls": 1.2095008632875088,
        }

    # Similarly, keep the `update_hypers` param here for API consistency
    def update_posterior(self, new_rating, new_z, update_hypers=False):

        # However, we don't want to actually change the hyper in any case
        return super().update_posterior(new_rating, new_z, update_hypers=False)

    def ends_training(self):
        # Overwrites super().ends_training(), because in the aggregated model,
        # we no longer wants to normalize the ratings.
        return

    def compute_utility(self, z):
        # Now that we don't normalize the rating, we won't be calling
        # compute_utility_inverse_normalized, we only need to compute the raw utility
        return self.compute_utility_raw(z)


def mse(x, y):
    return np.mean((x - y) ** 2)


def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def cor(x, y):
    return stats.pearsonr(x, y)[0]


def np_str(string_list):
    return np.array(eval(string_list))


train_ratings_all = train_df["ratings"].apply(np_str)
train_items_all = train_df["items_shown"].apply(np_str)
test_ratings_all = test_df["fixed_test_ratings"].apply(np_str)
test_items_all = test_df["fixed_test_items"].apply(np_str)


total_people = train_ratings_all.shape[0]
people_ix = np.arange(total_people)


gp_model = PrefOptimAgg(z_shown=None, ratings=None, opt_hypers=False)

new_rmse_inds = []
for i in trange(total_people):
    train_ratings = train_ratings_all.iloc[i]
    z_train = z_all[train_items_all.iloc[i]]
    test_ratings = test_ratings_all.iloc[i]
    z_test = z_all[test_items_all.iloc[i]]

    gp_model.x = z_train
    gp_model.y = train_ratings

    pred = gp_model.compute_utility(z=z_test)

    new_rmse_inds.append(rmse(pred, test_ratings))

argsort_id = np.argsort(new_rmse_inds)

good_people = people_ix[argsort_id[:5]]
bad_people = people_ix[argsort_id[-5:]]
random_people = np.random.choice(people_ix, 5)


umap = UMAP(random_state=0).fit(z_all)
z_tsned = umap.fit_transform(z_all)


def depict(i):
    train_ratings = train_ratings_all.iloc[i]
    z_train = z_all[train_items_all.iloc[i]]

    gp_model.x = z_train
    gp_model.y = train_ratings

    preds = gp_model.compute_utility(z_all)

    fig, ax = plt.subplots()
    sc = ax.scatter(x=z_tsned[:, 0], y=z_tsned[:, 1], c=preds, cmap="RdYlGn", alpha=0.8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = fig.colorbar(sc)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("Predicted utility", rotation=270)

    maxid = np.argmax(preds)
    minid = np.argmin(preds)

    return fig, ((maxid, z_tsned[maxid, 0], z_tsned[maxid, 1]), (minid, z_tsned[minid, 0], z_tsned[minid, 1]))


for ix, i in enumerate(good_people):
    fig, res = depict(i)
    fig.suptitle(
        f"UMAP projection of all items and GP-MU (Agg) models' predicted utilities.\n Individual with low testing RMSE."
    )
    fig.tight_layout()
    fig.savefig(f"results_analysis/reoptimize_backend_forward_looking/agg-visualize-embedding/good-{ix}.pdf")

    # print(f"Good {ix}, {i}")
    # print(res)
    # print()


for ix, i in enumerate(bad_people):
    fig, res = depict(i)
    fig.suptitle(
        f"UMAP projection of all items and GP-MU (Agg) models' predicted utilities.\n Individual with high testing RMSE."
    )
    fig.tight_layout()
    fig.savefig(f"results_analysis/reoptimize_backend_forward_looking/agg-visualize-embedding/bad-{ix}.pdf")

    print(f"Bad {ix}, {i}")
    print(res)
    print()

for ix, i in enumerate(random_people):
    fig, res = depict(i)
    fig.suptitle(
        f"UMAP projection of all items and GP-MU (Agg) models' predicted utilities.\n Randomly selected individual."
    )
    fig.tight_layout()
    fig.savefig(f"results_analysis/reoptimize_backend_forward_looking/agg-visualize-embedding/random-{ix}.pdf")

    # print(f"Random {ix}, {i}")
    # print(res)
    # print()

for i in range(z_tsned.shape[0]):
    x, y = z_tsned[i, 0], z_tsned[i, 1]

    if 10 <= x <= 11 and 7 <= y <= 8:
        print(i, x, y)


