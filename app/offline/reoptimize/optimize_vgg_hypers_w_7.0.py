import os

os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ["ADAPTIVE_PREFERENCE_ENV"] = "dev"

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from matplotlib import pyplot as plt
from tqdm import trange
from scipy.optimize import minimize
import re




# Import the PrefOptim object:

os.chdir(os.path.expanduser("~/ryandew_adaptive_preference/RA/results_analysis/reoptimize/"))
from og_reopt import PrefOptim


# Import useful analysis functions:

os.chdir(os.path.expanduser("~/ryandew_adaptive_preference/RA/results_analysis/reoptimize/"))
from utils import *


# Load the Z data

data_path = os.path.expanduser("~/ryandew_adaptive_preference/RA/backend/data/scaled_vgg_pca10_dresses.csv")
raw_data = pd.read_csv(data_path)

def create_pairs(dresses):
    dresses_cols = [col for col in dresses.columns if re.compile(r"E\d+").match(col)]  # Columns like E1, E2, ...
    indexed_dresses = dresses.reset_index(drop=False)  # dresses dataframe with indices ranging from 0 to N
    # Artificial column for catesian product. Pandas version is too low to use how="cross"
    indexed_dresses["_merge_key"] = 0
    merged_all_dresses = pd.merge(indexed_dresses, indexed_dresses, on="_merge_key", how="inner", suffixes=("_1", "_2"))
    merged_filtered_dresses = merged_all_dresses[merged_all_dresses["index_1"] < merged_all_dresses["index_2"]]
    for ix, col in enumerate(dresses_cols, 1):
        merged_filtered_dresses[f"Diff{ix}"] = merged_filtered_dresses[f"{col}_1"] - merged_filtered_dresses[f"{col}_2"]
    return merged_filtered_dresses[
        ["Item_ID_1", "Item_ID_2", "URL_1", "URL_2"] + [f"Diff{ix}" for ix, _ in enumerate(dresses_cols, 1)]
    ].reset_index(drop=True)

all_items = create_pairs(raw_data)
embedding_cols_start = 'Diff1'
z_all = all_items.loc[:, embedding_cols_start:].values


# Load the ratings data:

os.chdir(os.path.expanduser("~/ryandew_adaptive_preference/RA/results_analysis/reoptimize/"))
og_train = pd.read_csv(os.path.expanduser('~/Dropbox/1_proj/urbn/results/binary_mpc_results/analysis-7.0-knn1-binary-mpc-None/train_df.csv'))
og_test = pd.read_csv(os.path.expanduser('~/Dropbox/1_proj/urbn/results/binary_mpc_results/analysis-7.0-knn1-binary-mpc-None/test_df.csv'))

# Extract training
def extract_train(i):
    train_i = og_train.iloc[i]
    z_shown = z_all[np_str(train_i.items_shown)]
    # Note to self: ratings is the original ratings, no negation
    ratings = np_str(train_i.ratings)
    return z_shown, ratings

def extract_test(i):
    # Extract test
    test_i = og_test.iloc[i]
    random_test_i = eval(test_i['random_test_predictions_ratings'])
    test_items = list(random_test_i.keys())
    old_preds = np.array([a for (a,b) in list(random_test_i.values())])
    test_ratings = np.array([b for (a,b) in list(random_test_i.values())])
    return test_items, test_ratings, old_preds

def extract_sym(i):
    z_shown, ratings = extract_train(i)
    # Compute symmetric z/ratings
    sym_z = np.vstack([z_shown, -z_shown])
    sym_ratings = np.append(ratings, -ratings)
    return sym_z, sym_ratings

def fit_comparison_dict(test_ratings, pred_ratings1, pred_ratings2, index=0):
    return {
        'old_cor': cor(test_ratings, pred_ratings1),
        'new_cor': cor(test_ratings, pred_ratings2),
        'old_sign_acc': sign_acc(test_ratings, pred_ratings1),
        'new_sign_acc': sign_acc(test_ratings, pred_ratings2),
        'old_rmse': rmse(test_ratings, pred_ratings1),
        'new_rmse': rmse(test_ratings, pred_ratings2),
    }

def hyper_comparison_dict(model1, model2, index=0):
    return {
        'old_noise': model1.hypers['noise'],
        'old_amp': model1.hypers['amp'],
        'old_ls': model1.hypers['ls'],
        'new_noise': model2.hypers['noise'],
        'new_amp': model2.hypers['amp'],
        'new_ls': model2.hypers['ls'],
    }



## Optimization using RMSE ----------------------------------------------------------------------

def average_rmse(hypers_list, z_train_list, ratings_train_list, z_test_list, ratings_test_list):
    # Compute average rmse across all people
    tot_rmse = 0
    N = len(z_train_list)
    for i in range(N):
        prefs = PrefOptim(
            z_shown = z_train_list[i], 
            ratings = ratings_train_list[i]
        )
        prefs.hypers = {'noise': hypers_list[0], 'amp': hypers_list[1], 'ls': hypers_list[2]}
        pred = prefs.compute_utility(z_test_list[i])
        tot_rmse += rmse(pred, ratings_test_list[i])
    return tot_rmse / N

def optimize_average_rmse(pars0, z_train_list, ratings_train_list, z_test_list, ratings_test_list):
    bnds = ((1e-6, 100), (1e-6, 100), (1e-6, 100))
    opt_out = minimize(
        average_rmse, 
        pars0, 
        args=(z_train_list, ratings_train_list, z_test_list, ratings_test_list), 
        method="L-BFGS-B", 
        bounds=bnds
    )
    return {"noise": opt_out.x[0], "amp": opt_out.x[1], "ls": opt_out.x[2]}

# Compute optimal aggregate RMSE:
z_train_list = []
ratings_train_list = []
z_test_list = []
ratings_test_list = []
for i in range(og_train.shape[0]):
    z_sym, ratings_sym = extract_sym(i)
    test_items, test_ratings, old_preds = extract_test(i)
    z_train_list.append(z_sym)
    ratings_train_list.append(ratings_sym)
    z_test_list.append(z_all[test_items])
    ratings_test_list.append(test_ratings)

# agg_hypers = optimize_average_rmse(
#     pars0=[0.1,1,0.5], 
#     z_train_list=z_train_list, 
#     ratings_train_list=ratings_train_list, 
#     z_test_list=z_test_list, 
#     ratings_test_list=ratings_test_list
# )

# Parameters from a previous run:
agg_hypers = {'noise': 6.88756714098321, 'amp': 7.148767946480032, 'ls': 5.357894316288602}

average_rmse(
    [agg_hypers['noise'], agg_hypers['amp'], agg_hypers['ls']], 
    z_train_list, 
    ratings_train_list, 
    z_test_list, 
    ratings_test_list
)