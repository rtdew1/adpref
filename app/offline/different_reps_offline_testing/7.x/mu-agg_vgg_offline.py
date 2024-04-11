import os, sys, re

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["ADAPTIVE_PREFERENCE_ENV"] = "dev"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm


# SETTINGS - MANUAL ------------------------------------------------------

APP_VERSION = "Binary MPC - MU-Agg on Different Representations"
OUTPUT_FILE = "mu-agg_vgg_offline"
IS_MPC = True
target_dir = os.path.expanduser("~/ryandew_adaptive_preference/RA/results_analysis/different_reps_offline_testing/7.x/outputs/")
train_df = pd.read_csv(os.path.expanduser('~/Dropbox/1_proj/urbn/results/binary_mpc_results/analysis-7.3-aggregated-fixed-binary-mpc-None/train_df.csv'))
test_df = pd.read_csv(os.path.expanduser('~/Dropbox/1_proj/urbn/results/binary_mpc_results/analysis-7.3-aggregated-fixed-binary-mpc-None/test_df.csv'))
ind_df = pd.read_csv(os.path.expanduser('~/Dropbox/1_proj/urbn/results/binary_mpc_results/analysis-7.3-aggregated-fixed-binary-mpc-None/individuals.csv'))

data_files = [
    os.path.expanduser("~/ryandew_adaptive_preference/RA/backend/data/dresses10_urls.csv"),
    os.path.expanduser("~/ryandew_adaptive_preference/RA/backend/data/scaled_vgg_pca10_dresses.csv"),
]

data_names = ["Original", "VGG-PCA-10"]



# Get the correct PrefOptim class
os.chdir(os.path.expanduser('~/ryandew_adaptive_preference/RA/backend'))

from model.aggregated import PrefOptim




# RUN STUFF ------------------------------------------------------

# Exclude people who do not pass attention check:
train_df = train_df.loc[ind_df['passed_redisplay'] == 1,:]
test_df = test_df.loc[ind_df['passed_redisplay'] == 1,:]

## Silence printing
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

## Helper Function to format data for MPC studies.

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




## Generate data for Offline Testing

results = {}
data = {}
for d, DATA_FILE in enumerate(data_files):

    all_dresses = pd.read_csv(DATA_FILE)
    data[data_names[d]] = all_dresses
    if IS_MPC:
        all_items = create_pairs(all_dresses)
        embedding_cols_start = 'Diff1'
    else:
        all_items = all_dresses
        embedding_cols_start = 'E1'

    ## Train on varying number of dresses, computing RMSE for each individual, 
    ## then averaging over all individual for a specified number of training dresses.

    individuals = list(train_df.index)
    min_n_ratings = 5
    max_n_ratings = train_df.TOTAL_TRAINING_RATINGS.max()
    new_acc = []
    acc_df = pd.DataFrame(columns=['user_id'] + [str(i) for i in range(min_n_ratings, max_n_ratings+1)])
    for individual in individuals:
        acc_df.loc[len(acc_df.index)] = [individual] + [np.nan for i in range(min_n_ratings, max_n_ratings+1)]
    acc_df = acc_df.set_index("user_id").sort_index()

    # Here the min is used, but this may lead to weird behavior 


    for i in tqdm(individuals):
        if i in test_df.index:
            with HiddenPrints():
                ## All of i's training items and ratings
                all_items_shown = eval(train_df.loc[i].loc['items_shown'])
                all_train_y = np.array(eval(train_df.loc[i].loc['ratings']))

                ## Items used for back testing
                random_test_result_dict = eval(test_df.loc[i, 'random_test_predictions_ratings'])
                random_test_items = list(random_test_result_dict.keys())
                test_X = np.array(all_items.iloc[random_test_items].loc[:, embedding_cols_start:].values)
                y_actual = np.array([random_test_result_dict[item][1] for item in random_test_items])

                if min_n_ratings > (all_train_y.shape[0] + 1):
                    continue
                else:
                    for k in range(min_n_ratings, all_train_y.shape[0] + 1):
                        train_items = all_items_shown[:k]
                        train_X = np.array(all_items.iloc[train_items].loc[:, embedding_cols_start:].values)
                        train_y = all_train_y[:k]

                        if IS_MPC:
                            mirror_X = -train_X
                            mirror_y = -train_y
                            train_X = np.vstack((train_X, mirror_X))
                            train_y = np.concatenate((train_y, mirror_y))

                        ## Train model
                        model = PrefOptim(train_X, train_y)
                        if data_names[d] == "VGG-PCA-10":
                            model.hypers = {'noise': 4.105707235134857, 'amp': 2.594658008010263, 'ls': 1.158389347265231}
                        new_y_pred = model.compute_utility(test_X)

                        ## Compute Individual RMSE and Correlation
                        correct_predictions = [1 if (np.sign([new_y_pred[j]]) == np.sign([y_actual[j]]) or (y_actual[j] == 0.0)) 
                                                    else 0 for j in range(len(y_actual))]
                        new_acc_i = np.mean(correct_predictions)
                        acc_df.loc[[i], str(k)] = new_acc_i

    acc_by_ratings = np.nanmean(acc_df.values, axis=0)
    results[data_names[d]] = {'acc_df': acc_df, 'acc_by_ratings': acc_by_ratings}

# Create target_dir if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

results_df = pd.DataFrame({data_name: results[data_name]['acc_by_ratings'] for data_name in data_names})
results_df.to_csv(os.path.join(target_dir, f"{OUTPUT_FILE}.csv"), index=False)

for data_name in data_names:
    acc_df = results[data_name]['acc_df']
    acc_df.to_csv(os.path.join(target_dir, f"{OUTPUT_FILE}__{data_name}_acc_df.csv"), index=False)