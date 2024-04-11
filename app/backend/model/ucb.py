# IMPORTS -------------------------------------------

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from model import gp_inf, gp_inf_fix_amp
from model.og import PrefOptim as OG


## MODEL --------------------------------------------------------


class PrefOptim(OG):
    def acquisition_fn(self, z):
        return self.ucb(z, scaling_factor=2.0)
