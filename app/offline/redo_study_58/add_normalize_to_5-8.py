import os

os.environ["ADAPTIVE_PREFERENCE_ENV"] = "dev"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.expanduser('~/ryandew_adaptive_preference/RA/backend'))

from model.gur import PrefOptim

def unit_constrain(z: np.ndarray, padding=1e-4):
    return padding + (1 - 2 * padding) * (z - z.min()) / (z.max() - z.min())

# Load data
item_data_file = os.path.expanduser('~/ryandew_adaptive_preference/RA/backend/data/dresses10_urls.csv')
all_items = pd.read_csv(item_data_file)

# Compute constrained z representations
z_unconstr_all = np.array(all_items.loc[:, "E1":].values)
z_all = np.apply_along_axis(unit_constrain, 0, z_unconstr_all)

train_df = pd.read_csv(os.path.expanduser('~/ryandew_adaptive_preference/RA/results_analysis/redo_study_58/5-8_train_df.csv'))
test_df = pd.read_csv(os.path.expanduser('~/ryandew_adaptive_preference/RA/results_analysis/redo_study_58/5-8_test_df.csv'))
redone_test_df = test_df.copy()

cors = []
old_cors = []

for i in range(len(train_df)):
    # Recreate model
    items_shown = eval(train_df.iloc[i].loc['items_shown'])
    z_shown = np.array(z_all[items_shown])
    ratings = np.array(eval(train_df.iloc[i].loc['ratings']))
    hypers = eval(train_df.iloc[i].hypers)

    user_prefs = PrefOptim(z_shown, ratings, opt_hypers=False)
    user_prefs.hypers = hypers
    user_prefs.ends_training()

    renorm_train_pred = user_prefs.compute_utility(z_shown)

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

    renorm_random_test_pred = user_prefs.compute_utility(z_random_test)
    renorm_good_test_pred = user_prefs.compute_utility(z_good_test)

    redone_test_df.loc[i,"random_test_predictions_ratings"] = str(dict(zip(random_test_items, zip(renorm_random_test_pred, random_test_ratings)))) 
    redone_test_df.loc[i,"good_test_predictions_ratings"] = str(dict(zip(good_test_items, zip(renorm_good_test_pred, good_test_ratings)))) 
    redone_test_df.loc[i,"hypers"] = str(user_prefs.hypers)

    cors.append(np.corrcoef(renorm_random_test_pred, random_test_ratings)[0,1])
    old_cors.append(np.corrcoef(random_test_orig_preds, random_test_ratings)[0,1])

plt.hist(np.array(cors))
plt.hist(np.array(old_cors))

redone_test_df.to_csv(os.path.expanduser('~/ryandew_adaptive_preference/RA/results_analysis/redo_study_58/5-8_redone_test_df.csv'), index=False)