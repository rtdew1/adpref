import os, sys, re

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["ADAPTIVE_PREFERENCE_ENV"] = "dev"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

# SETTINGS - MANUAL ------------------------------------------------------

APP_VERSION = "Binary MPC - GUR"
FILE_EXTENSION = "gur"
DATA_FILE = os.path.expanduser("~/ryandew_adaptive_preference/RA/backend/data/dresses10_urls.csv")
IS_MPC = True
target_dir = os.path.expanduser("~/ryandew_adaptive_preference/RA/results_analysis/n_ratings_offline_testing/7.x_acc/gur/")
train_df = pd.read_csv(os.path.expanduser('~/Dropbox/1_proj/urbn/results/binary_mpc_results/analysis-7.8-gur_approx_sym-binary-mpc-update1-new_cand-None/train_df.csv'))
test_df = pd.read_csv(os.path.expanduser('~/Dropbox/1_proj/urbn/results/binary_mpc_results/analysis-7.8-gur_approx_sym-binary-mpc-update1-new_cand-None/test_df.csv'))
ind_df = pd.read_csv(os.path.expanduser('~/Dropbox/1_proj/urbn/results/binary_mpc_results/analysis-7.8-gur_approx_sym-binary-mpc-update1-new_cand-None/individuals.csv'))

# Get the correct PrefOptim class
os.chdir(os.path.expanduser('~/ryandew_adaptive_preference/RA/backend'))

from model.gur_sym import PrefOptim



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

all_dresses = pd.read_csv(DATA_FILE)
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
                    new_y_pred = model.compute_utility(test_X)

                    ## Compute Individual RMSE and Correlation
                    correct_predictions = [1 if (np.sign([new_y_pred[j]]) == np.sign([y_actual[j]]) or (y_actual[j] == 0.0)) 
                                                else 0 for j in range(len(y_actual))]
                    new_acc_i = np.mean(correct_predictions)
                    acc_df.loc[[i], str(k)] = new_acc_i



acc_by_ratings = np.nanmean(acc_df.values, axis=0)

plt.figure(1)
plt.plot(np.arange(min_n_ratings, max_n_ratings + 1), acc_by_ratings)
plt.xlabel('Training Size')
plt.ylabel('Test Accuracy')
plt.title(f'{APP_VERSION}')
acc_plot_path = os.path.join(target_dir, f"{FILE_EXTENSION}-acc.png")
plt.savefig(acc_plot_path)

acc_path = os.path.join(target_dir, f"{FILE_EXTENSION}-individual_acc_df.csv")
acc_df.to_csv(acc_path)

