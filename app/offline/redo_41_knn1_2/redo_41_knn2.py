import os, sys

os.environ["ADAPTIVE_PREFERENCE_ENV"] = "dev"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.expanduser('~/ryandew_adaptive_preference/RA/backend'))

from model.base import PrefOptimBase
from sklearn.neighbors import KNeighborsRegressor

class PrefOptim(PrefOptimBase):
    def __init__(self, X, y, opt_hypers=None, map=False):
        super().__init__()
        print("model: knn")

        self.version = "knn-k2-redo"
        self.K = 2
        self.knnmodel = KNeighborsRegressor(n_neighbors=self.K, weights="distance")
        self.X = X
        self.y = y

    # update_hypers and map are intentionally left here for API consistency
    def update_posterior(self, y_new, x_new, update_hypers=False, map=False):
        self.X = np.append(self.X, np.reshape(x_new, (-1, 1)).T, axis=0)
        self.y = np.append(self.y, y_new)

    def next_item(self, Z):
        all_indices = np.arange(Z.shape[0])
        np.random.shuffle(all_indices)
        return all_indices

    def compute_utility_raw(self, z):
        if len(self.y) >= self.K:
            self.knnmodel.fit(self.X, self.y)
            predicted = self.knnmodel.predict(z)
        else:
            predicted = np.mean(self.y)
        return predicted

    @property
    def hypers(self):
        """
        In the real model (model.py), the hyperparameters are assumed to be updating each time.
        The app.py will raise an error if such updating is not observed. To meet with such requirement,
        the dummy attribute `hyper` in this KNNPrefOptim returns a dictionary with random values each time invoked.
        """
        return {"dummy": np.random.rand()}

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def unit_constrain(z: np.ndarray, padding=1e-4):
    return padding + (1 - 2 * padding) * (z - z.min()) / (z.max() - z.min())

# Load data
item_data_file = os.path.expanduser('~/ryandew_adaptive_preference/RA/backend/data/dresses10_urls.csv')
all_items = pd.read_csv(item_data_file)

# Compute constrained z representations
z_unconstr_all = np.array(all_items.loc[:, "E1":].values)
z_all = np.apply_along_axis(unit_constrain, 0, z_unconstr_all)

train_df = pd.read_csv(os.path.expanduser('~/ryandew_adaptive_preference/RA/results_analysis/redo_41_knn1_2/4-1_train_df.csv'))
test_df = pd.read_csv(os.path.expanduser('~/ryandew_adaptive_preference/RA/results_analysis/redo_41_knn1_2/4-1_test_df.csv'))
redone_test_df = test_df.copy()

with HiddenPrints():
    for i in range(len(train_df)):
        # Recreate model
        items_shown = eval(train_df.iloc[i].loc['items_shown'])
        z_shown = np.array(z_all[items_shown])
        ratings = np.array(eval(train_df.iloc[i].loc['ratings']))

        user_prefs = PrefOptim(z_shown, ratings)
        user_prefs.K = 2
        user_prefs.ends_training()

        new_train_pred = user_prefs.compute_utility(z_shown)

        random_test_result_dict = eval(test_df.iloc[i].loc['random_test_predictions_ratings'])
        random_test_items = list(random_test_result_dict.keys())
        random_test_orig_preds = [random_test_result_dict[item][0] for item in random_test_items]
        random_test_ratings = [random_test_result_dict[item][1] for item in random_test_items]
        z_random_test = np.array(z_all[random_test_items])

        good_test_result_dict = eval(test_df.iloc[i].loc['good_test_predictions_ratings'])
        good_test_items = list(good_test_result_dict.keys())
        good_test_orig_preds = [good_test_result_dict[item][0] for item in good_test_items]
        good_test_ratings = [good_test_result_dict[item][1] for item in good_test_items]
        z_good_test = np.array(z_all[good_test_items])

        new_random_test_pred = user_prefs.compute_utility(z_random_test)
        new_good_test_pred = user_prefs.compute_utility(z_good_test)

        redone_test_df.loc[i,"random_test_predictions_ratings"] = str(dict(zip(random_test_items, zip(new_random_test_pred, random_test_ratings)))) 
        redone_test_df.loc[i,"good_test_predictions_ratings"] = str(dict(zip(good_test_items, zip(new_good_test_pred, good_test_ratings)))) 

redone_test_df.loc[:,"model_version"] = "knn-k2-redo"
redone_test_df.to_csv(os.path.expanduser('~/ryandew_adaptive_preference/RA/results_analysis/redo_41_knn1_2/4-1_knn2_test_df.csv'), index=False)