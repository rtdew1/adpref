from model.og import PrefOptim as OG
import numpy as np


class PrefOptim(OG):

    # Keep the `opt_hypers` params here for API consistency
    def __init__(self, z_shown, ratings, opt_hypers=True, map=True):

        # However, when initializing, we are not actually doing optimization
        super().__init__(z_shown, ratings, opt_hypers=False, map=map)

        print("using aggregated hyperparameters")

        # Instead, use the hyper values computed from old results:
        # These are on the original dresses data, using cor as the metric:
        # self.hypers = {
        #     'noise': 0.38621058403938996,
        #     'amp': 1.905899429488762,
        #     'ls': 3.577134415736614
        # }

        # These were optimized on the scaled VGG data, using cor as the metric:
        self.hypers = {'noise': 4.105707235134857, 'amp': 2.594658008010263, 'ls': 1.158389347265231}


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
