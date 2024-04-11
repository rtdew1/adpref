import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from model import gp_inf, gp_inf_fix_amp
from model.base import PrefOptimBase
from time import time


class PrefOptim(PrefOptimBase):
    def __init__(self, z_shown, ratings, opt_hypers=True, map=True):
        super().__init__()
        print("model: symmetric gur")

        self.version = "symgur1.0"
        self.x = z_shown
        self.y = ratings
        self.map = map

        initial_hypers = [0.2, 1, 1]
        # optim_args = {"x": self.x, "y": self.y}
        # if opt_hypers:
        #     if map:
        #         self.hypers = gp_inf.map_hypers(initial_hypers, optim_args)
        #     else:
        #         self.hypers = gp_inf.optim_hypers(initial_hypers, optim_args)
        # else:
        #    self.hypers = initial_hypers
        self.hypers = {
            "noise": initial_hypers[0],
            "amp": initial_hypers[1],
            "ls": initial_hypers[2],
        }
        print(self.hypers)

    def compute_new_var(self, z_target, z_other):
        var1 = gp_inf.compute_gp_var(
            new_x=z_other,
            x=np.vstack([self.x, z_target]),
            hypers=self.hypers,
        ).sum()
        return var1

    def gur(self, z):
        # Compute the sum of all uncertainty with the current data "self.x"
        var0 = gp_inf.compute_gp_var(new_x=z, x=self.x, hypers=self.hypers).sum()
        # For each potential new data point in z, compute how much the variance goes down by querying z
        return np.array(
            [
                var0 - self.compute_new_var(z[i], np.delete(z, i, axis=0))
                for i in range(z.shape[0])
            ]
        )

    def gur_sym(self, z):
        gur_z = self.gur(z)
        for i in range(z.shape[0]):
            # Assume you have already observed z[i]:
            self.x = np.vstack([self.x, z[i]])
            # Remaining variance after observing z[i]:
            var0 = gp_inf.compute_gp_var(
                new_x=np.delete(z, i, axis=0), x=self.x, hypers=self.hypers
            ).sum()
            # GUR after observing -z[i]:
            gur_z[i] += var0 - self.compute_new_var(-z[i], np.delete(z, i, axis=0))
            # Remove the "fake" observation of z[i] from self.x:
            self.x = self.x[:-1, :]
        return gur_z

    def gur_approx(self, z):
        var = gp_inf.compute_gp_var(
            new_x=z,
            x=self.x,
            hypers=self.hypers,
        )
        resid_var = np.sqrt(var) - self.hypers["noise"]
        K = gp_inf.K(z, z, self.hypers)
        gur_approx = (K * resid_var).sum(1)
        return gur_approx

    def gur_approx_subsample(
        self, z, num_cands=1000, num_cands_total=10000, num_eval_points=10000
    ):
        var = gp_inf.compute_gp_var(
            new_x=z,
            x=self.x,
            hypers=self.hypers,
        )
        resid_var = np.sqrt(var) - self.hypers["noise"]
        cands = np.random.choice(
            np.argsort(resid_var)[-num_cands_total:], num_cands, replace=False
        )
        eval_points = np.random.choice(z.shape[0], num_eval_points, replace=False)
        K = gp_inf.K(z[cands], z[eval_points], self.hypers)
        gur_approx_cands = (K * resid_var[eval_points]).sum(1)
        gur_approx = np.zeros(z.shape[0])
        gur_approx[cands] = gur_approx_cands
        return gur_approx

    def gur_approx_sym(self, z):
        gur = self.gur_approx(z)
        for i in range(z.shape[0]):
            self.x = np.vstack([self.x, z[i]])
            var = gp_inf.compute_gp_var(
                new_x=np.delete(z, i, axis=0),
                x=self.x,
                hypers=self.hypers,
            )
            resid_var = np.sqrt(var) - self.hypers["noise"]
            K = gp_inf.K(
                np.delete(z, i, axis=0), -z[i].reshape(1, -1), self.hypers
            ).flatten()
            gur[i] += (K * resid_var).sum()
            self.x = self.x[:-1, :]
        return gur

    # THIS IS THE OLD VERSION; IT ONLY WORKS WITH LIKE 20 CANDIDATES
    # def gur_approx_mu_heur_sym(self, z, num_cand=2000):
    #     if (num_cand is not None) and (z.shape[0] > num_cand):
    #         # Compute original amount of variance:
    #         var0 = gp_inf.compute_gp_var(
    #             new_x=z,
    #             x=self.x,
    #             hypers=self.hypers,
    #         )
    #         resid_var0 = np.sqrt(var0) - self.hypers["noise"]

    #         # For each candidate point, compute the GUR approximation
    #         candidates = np.argsort(resid_var0)[-num_cand:]
    #         K0 = gp_inf.K(z[candidates], z[candidates], self.hypers)
    #         approx_gur_cands = (K0 * resid_var0[candidates]).sum(1)

    #         # Consider each candidate's symmetric point:
    #         for j in range(num_cand):
    #             i = candidates[j]
    #             self.x = np.vstack([self.x, z[i]])
    #             var1 = gp_inf.compute_gp_var(
    #                 new_x=np.delete(z, i, axis=0),
    #                 x=self.x,
    #                 hypers=self.hypers,
    #             )
    #             resid_var1 = np.sqrt(var1) - self.hypers["noise"]
    #             K1 = gp_inf.K(
    #                 np.delete(z, i, axis=0), -z[i].reshape(1, -1), self.hypers
    #             ).flatten()
    #             approx_gur_cands[j] += (K1 * resid_var1).sum()
    #             self.x = self.x[:-1, :]

    #         approx_gur = np.zeros(z.shape[0])
    #         approx_gur[candidates] = approx_gur_cands
    #         return approx_gur

    #     else:
    #         return self.gur_approx(z)

    # def gur_approx_subsample_sym(
    #     self, z, num_cands=50, num_cands_total=5000, num_eval_points=5000
    # ):
    #     var = gp_inf.compute_gp_var(
    #         new_x=z,
    #         x=self.x,
    #         hypers=self.hypers,
    #     )
    #     resid_var = np.sqrt(var) - self.hypers["noise"]
    #     cands = np.random.choice(
    #         np.argsort(resid_var)[-num_cands_total:], num_cands, replace=False
    #     )
    #     eval_points = np.random.choice(len(resid_var), num_eval_points, replace=False)
    #     K = gp_inf.K(z[cands], z[eval_points], self.hypers)
    #     gur_approx_cands = (K * resid_var[eval_points]).sum(1)
    #     gur_approx = np.zeros(z.shape[0])
    #     gur_approx[cands] = gur_approx_cands

    #     # Symmetric step:
    #     for i in cands:
    #         self.x = np.vstack([self.x, z[i]])
    #         # Recompute variance, assuming we know z[i]:
    #         var = gp_inf.compute_gp_var(
    #             new_x=np.delete(z, i, axis=0),
    #             x=self.x,
    #             hypers=self.hypers,
    #         )
    #         resid_var = np.sqrt(var) - self.hypers["noise"]
    #         sym_eval_points = np.random.choice(
    #             len(resid_var), num_eval_points, replace=False
    #         )
    #         # Compute kernel between negative z[i] and all eval points:
    #         K = gp_inf.K(
    #             -z[i].reshape(1, -1),
    #             np.delete(z, i, axis=0)[sym_eval_points],
    #             self.hypers,
    #         ).flatten()

    #         gur_approx[i] += (K * resid_var[sym_eval_points]).sum()
    #         self.x = self.x[:-1, :]

    #     return gur_approx

    def gur_approx_subsample_sym(
        self, z, num_cands=1000, num_cands_total=1000, num_eval_points=20000
    ):
        # Normalize the amplitude to 1 so that multiplying by K always yields <= 1
        hypers = self.hypers
        hypers["amp"] = 1.0

        # Set eval points, to be used in both stages
        if num_eval_points > z.shape[0]:
            num_eval_points = z.shape[0]
        eval_points = np.random.choice(z.shape[0], num_eval_points, replace=False)

        # Compute original variance
        var = gp_inf.compute_gp_var(
            new_x=z,
            x=self.x,
            hypers=self.hypers,
        )
        resid_var = np.sqrt(var) - self.hypers["noise"]

        # Randomly select candidate points
        if num_cands < num_cands_total:
            cands = np.random.choice(
                np.argsort(resid_var)[-num_cands_total:], num_cands, replace=False
            )
        elif num_cands == num_cands_total:
            cands = np.argsort(resid_var)[-num_cands_total:]
        else:
            raise Exception("num_cands cannot be more than num_cands_total")
        K = gp_inf.K(z[cands], z[eval_points], self.hypers)

        # Original resid var at eval points:
        resid_var_eval = resid_var[eval_points]

        # Compute estimated reduction:
        uncert_reduc = K * resid_var_eval

        # Update resid var:
        resid_var_eval = resid_var_eval - uncert_reduc

        # Symmetric calculation:
        K_sym = gp_inf.K(-z[cands], z[eval_points], self.hypers)
        sym_uncert_reduc = K_sym * resid_var_eval

        # Total UR
        gur_approx_cands = (uncert_reduc + sym_uncert_reduc).sum(1)
        gur_approx = np.zeros(z.shape[0])
        gur_approx[cands] = gur_approx_cands

        return gur_approx

    def acquisition_fn(self, z):
        af = self.gur_approx_subsample_sym(z)
        return af

    def update_posterior(self, new_rating, new_z, update_hypers=False):
        print(self.hypers)
        print("rating:", new_rating)
        print("z:", new_z)
        self.x = np.vstack([self.x, new_z])
        self.y = np.concatenate((self.y, new_rating))
        if update_hypers:
            print("updating hypers")
            optim_args = {"x": self.x, "y": self.y}
            initial_hypers = [0.1, 1, 0.1]
            if self.map:
                self.hypers = gp_inf.map_hypers(initial_hypers, optim_args)
            else:
                self.hypers = gp_inf.optim_hypers(initial_hypers, optim_args)

    def next_item(self, available_z):
        ranking = np.argsort(self.acquisition_fn(available_z))[::-1]
        return ranking

    def compute_utility_raw(self, z):
        return gp_inf.compute_gp(
            new_x=z, x=self.x, y=self.y, hypers=self.hypers, compute_cov=False
        )

    def ends_training(self):
        super().ends_training()  # self.y has been normalized here in the super call

        initial_hypers = [0.1, 0.8]
        optim_args = {"x": self.x, "y": self.y}
        if self.map:
            self.hypers = gp_inf_fix_amp.map_hypers(initial_hypers, optim_args)
        else:
            self.hypers = gp_inf_fix_amp.optim_hypers(initial_hypers, optim_args)
