# Adaptive Design of Robust Choice Questionnaires

from model.base import PrefOptimBase
import numpy as np
import scipy.linalg
from sklearn.metrics.pairwise import cosine_similarity


class PrefOptim(PrefOptimBase):
    def __init__(self, z_shown, ratings, opt_hypers=True, map=True):
        print("model: abernethy")
        super().__init__()
        self.x = z_shown
        self.y = ratings
        self.version = 1.2
        self._update_internal_params()

    def _update_internal_params(self):
        self.lambda_ = 1 / self.x.shape[0]
        pen = np.identity(self.x.shape[1]) * self.lambda_
        invK = np.linalg.inv(self.x.T @ self.x + pen)
        self.beta_fit = invK @ self.x.T @ self.y

    def update_posterior(self, new_rating, new_z, update_hypers=False):
        self.x = np.vstack([self.x, new_z])
        self.y = np.concatenate((self.y, new_rating))
        if update_hypers:
            self._update_internal_params()

    def next_item(self, available_z):
        pen_d = np.identity(self.x.shape[1]) * self.lambda_
        Iww = np.identity(self.x.shape[1]) - ((self.beta_fit @ self.beta_fit.T) / (self.beta_fit.T @ self.beta_fit))
        HessMatrix = Iww @ (self.x.T @ self.x + pen_d)

        eig_val, eig_vec = scipy.linalg.eig(HessMatrix)

        sort_order = [
            (
                # Sort from small to large, so 0 means prioritized.
                # Prioritize real eigenvalue > 0
                0 if (np.real(e) > 0 and np.imag(e) == 0) else 1,
                e,
            )
            for e in eig_val
        ]

        chosen_eig_ix = np.argmin(sort_order)
        ideal_point = eig_vec[:, chosen_eig_ix]

        abs_cos_sims = np.abs(cosine_similarity(available_z, np.real(ideal_point).reshape(1, -1))).reshape(-1)
        return np.argsort(abs_cos_sims)[::-1]

    def compute_utility_raw(self, z):
        return z @ self.beta_fit

    def ends_training(self):
        super().ends_training()  # self.y has been normalized here in the super call
        self._update_internal_params()

    @property
    def hypers(self):
        return {
            "lambda": self.lambda_,
            # Storing all beta takes up a lot of space,
            # only return the first one to confirm that the hypers are updating.
            "beta0": self.beta_fit[0],
        }
