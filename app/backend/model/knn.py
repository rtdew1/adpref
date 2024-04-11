## IMPORTS ------------------------------------------------------

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from model.base import PrefOptimBase


## MODEL --------------------------------------------------------


class PrefOptim(PrefOptimBase):
    def __init__(self, X, y, opt_hypers=None, map=False):
        super().__init__()
        print("model: knn")

        self.version = 1.2
        self.K = 1
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
