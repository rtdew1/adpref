import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from model import gp_inf, gp_inf_fix_amp
from model.base import PrefOptimBase


class PrefOptim(PrefOptimBase):
    def __init__(
        self,
        z_shown,
        ratings,
        opt_hypers=True,
        map=True,
        init=[0.1, 0.4, 0.8], # Changed from [0.1, 1, 0.1]
        addl_optim_args={
            'noise_prior': {'a': 5, 'scale': 0.05},
            'ls_prior': {'a': 14, 'scale': 14},
            'amp_prior': {'a': 12, 'scale': 3}
        },
    ):
        super().__init__()

        self.version = "gur-ar-2.0"
        self.x = z_shown
        self.y = ratings
        self.map = map
        self.init = init
        self.addl_optim_args = addl_optim_args

        initial_hypers = init

        optim_args = {
            "x": self.x,
            "y": self.y,
        }
        if self.addl_optim_args is not None:
            optim_args.update(self.addl_optim_args)

        if opt_hypers:
            if map:
                self.hypers = gp_inf.map_hypers(initial_hypers, optim_args)
            else:
                self.hypers = gp_inf.optim_hypers(initial_hypers, optim_args)

        else:
            self.hypers = initial_hypers

    def gur(self, z):
        var0 = gp_inf.compute_gp_var(new_x=z, x=self.x, hypers=self.hypers).sum()

        def compute_new_var(z_target, z_other):
            var1 = gp_inf.compute_gp_var(
                new_x=z_other,
                x=np.vstack([self.x, z_target]),
                hypers=self.hypers,
            ).sum()
            return var1

        gur = np.array(
            [
                var0 - compute_new_var(z[i], np.delete(z, i, axis=0))
                for i in range(z.shape[0])
            ]
        )
        return gur

    def gur_approx(self, z, num_cand=None):
        """
        Compute an approximation to GUR. If num_cand (Int) is specified, use the MU heuristic to
        select candidate points. This is useful if the number of possible points is very large.
        The MU heuristic is only applied if there are more possible points than num_cand.
        """
        var0 = gp_inf.compute_gp_var(
            new_x=z,
            x=self.x,
            hypers=self.hypers,
        )
        # Approximate the amount of variance reduced at an observation point by the original variance
        # minus the noise, which we're terming the residual variance (after noise):
        resid_var = np.sqrt(var0) - self.hypers["noise"]
        # If using the MU approximation:
        if (num_cand is not None) and (z.shape[0] > num_cand):
            candidates = np.argsort(resid_var)[-num_cand:]
            K = gp_inf.K(z[candidates], z[candidates], self.hypers)
            gur_approx_cands = (K * resid_var[candidates]).sum(1)
            # GUR is set to zero for non-candidate points:
            gur_approx = np.zeros(z.shape[0])
            gur_approx[candidates] = gur_approx_cands
        else:
            K = gp_inf.K(z, z, self.hypers)
            gur_approx = (K * resid_var).sum(1)
        return gur_approx

    def gur_approx_subsample(self, z, num_cands=1000, num_cands_total=10000, num_eval_points=10000):
        var = gp_inf.compute_gp_var(
            new_x=z,
            x=self.x,
            hypers=self.hypers,
        )
        resid_var = np.sqrt(var) - self.hypers['noise']
        cands = np.random.choice(np.argsort(resid_var)[-num_cands_total:], num_cands, replace=False)
        eval_points = np.random.choice(z.shape[0], num_eval_points, replace=False)
        K = gp_inf.K(z[cands], z[eval_points], self.hypers)
        gur_approx_cands = (K * resid_var[eval_points]).sum(1)
        gur_approx = np.zeros(z.shape[0])
        gur_approx[cands] = gur_approx_cands
        return gur_approx
    
    def acquisition_fn(self, z):
        return self.gur_approx(z, num_cand=500)

    def update_posterior(self, new_rating, new_z, update_hypers=False):
        self.x = np.vstack([self.x, new_z])
        self.y = np.concatenate((self.y, new_rating))
        if update_hypers:
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
