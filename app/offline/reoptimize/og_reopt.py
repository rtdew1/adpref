# IMPORTS -------------------------------------------

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import gp_inf_reopt as gp_inf
import gp_inf_fix_amp_reopt as gp_inf_fix_amp
from base import PrefOptimBase


## MODEL --------------------------------------------------------


class PrefOptim(PrefOptimBase):
    def __init__(self, z_shown, ratings, opt_hypers=True, map=True, init=[0.1,0.4,0.8], addl_optim_args=None):
        super().__init__()

        self.version = 2.0
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

    def ucb(self, z, scaling_factor):
        gp_mean, gp_cov = gp_inf.compute_gp(new_x=z, x=self.x, y=self.y, hypers=self.hypers)
        sigma = 2 * np.sqrt(gp_cov)
        return gp_mean + scaling_factor * sigma

    def info_gain(self, z):
        gp_cov = gp_inf.compute_gp(new_x=z, x=self.x, y=self.y, hypers=self.hypers)[1]
        return 2 * np.sqrt(gp_cov)

    def acquisition_fn(self, z):
        return self.info_gain(z)

    def update_posterior(self, new_rating, new_z, update_hypers=False):
        self.x = np.vstack([self.x, new_z])
        self.y = np.concatenate((self.y, new_rating))
        if update_hypers:
            optim_args = {
                "x": self.x, 
                "y": self.y, 
            }
            if self.addl_optim_args is not None:
                optim_args.update(self.addl_optim_args)
            initial_hypers = self.init
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
