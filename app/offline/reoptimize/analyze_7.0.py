import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

os.environ['ADAPTIVE_PREFERENCE_ENV'] = 'dev'

# Step 1: Load the Z data
os.chdir(os.path.expanduser("~/ryandew_adaptive_preference/RA/backend/"))
from data_io.mpc.dataset import z_all

# Step 2: Import the PrefOptim object:
os.chdir(os.path.expanduser("~/ryandew_adaptive_preference/RA/backend/"))
from model.og import PrefOptim

# Step 3: Import useful analysis functions:
os.chdir(os.path.expanduser("~/ryandew_adaptive_preference/RA/results_analysis/reoptimize_mpc/"))
from utils import *


## BEGIN ANALYSIS ------------------------------

os.chdir(os.path.expanduser("~/ryandew_adaptive_preference/RA/results_analysis/reoptimize_mpc/"))
og_train = pd.read_csv("inputs/7.0_knn_binary_train_df.csv")
og_test = pd.read_csv("inputs/7.0_knn_binary_test_df.csv")

# Note to self: ratings is the original ratings, no negation

i = 0

train_i = og_train.iloc[i]
z_shown = z_all[np_str(train_i.items_shown)]
ratings = np_str(train_i.ratings)

test_i = og_test.iloc[i]
random_test_i = eval(test_i['random_test_predictions_ratings'])
test_items = list(random_test_i.keys())
old_preds = np.array([a for (a,b) in list(random_test_i.values())])
test_ratings = np.array([b for (a,b) in list(random_test_i.values())])

new_preds = PrefOptim(
    z_shown = z_shown,
    ratings = ratings,
).compute_utility(z_all[test_items])

plt.scatter(old_preds, new_preds)

sym_z = np.vstack([z_shown, -z_shown])
sym_ratings = np.append(ratings, -ratings)

sym_preds = PrefOptim(
    z_shown = sym_z,
    ratings = sym_ratings,
).compute_utility(z_all[test_items])

plt.figure()
plt.scatter(old_preds, sym_preds)
plt.show()

plt.figure()
plt.scatter(new_preds, sym_preds)
plt.show()

print("cor, old: ", cor(test_ratings, old_preds))
print("cor, new, single: ", cor(test_ratings, new_preds))
print("cor, new, sym: ", cor(test_ratings, sym_preds))

print("sign_acc, old: ", sign_acc(test_ratings, old_preds))
print("sign_acc, new, single: ", sign_acc(test_ratings, new_preds))
print("sign_acc, new, sym: ", sign_acc(test_ratings, sym_preds))

print("rmse, old: ", rmse(test_ratings, old_preds))
print("rmse, new, single: ", rmse(test_ratings, new_preds))
print("rmse, new, sym: ", rmse(test_ratings, sym_preds))