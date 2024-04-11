import os

os.environ["ADAPTIVE_PREFERENCE_ENV"] = "dev"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import scipy.stats as stats
import operator

os.chdir(os.path.expanduser('~/ryandew_adaptive_preference/RA/backend'))

from model.knn import PrefOptim

def unit_constrain(z: np.ndarray, padding=1e-4):
    return padding + (1 - 2 * padding) * (z - z.min()) / (z.max() - z.min())

# Load data
item_data_file = os.path.expanduser('~/ryandew_adaptive_preference/RA/backend/data/dresses10_urls.csv')
all_dresses = pd.read_csv(item_data_file)

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

all_items = create_pairs(all_dresses)
z_all = np.array(all_items.loc[:, "Diff1":].values)

train_df = pd.read_csv(os.path.expanduser('~/ryandew_adaptive_preference/RA/results_analysis/analysis-6.2-knn-mpc-dresses10-None/train_df.csv'))
test_df = pd.read_csv(os.path.expanduser('~/ryandew_adaptive_preference/RA/results_analysis/analysis-6.2-knn-mpc-dresses10-None/test_df.csv'))
redone_test_df = test_df.copy()

for i in range(len(train_df)):
    # Recreate model
    items_shown = eval(train_df.iloc[i].loc['items_shown'])
    z_shown = np.array(z_all[items_shown])
    ratings = np.array(eval(train_df.iloc[i].loc['ratings']))

    train_X = z_shown
    train_y = ratings

    mirror_X = -train_X
    mirror_y = -train_y
    train_X = np.vstack((train_X, mirror_X))
    train_y = np.concatenate((train_y, mirror_y))
    
    user_prefs = PrefOptim(train_X, train_y)
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

train_df.to_csv(os.path.expanduser('~/ryandew_adaptive_preference/RA/results_analysis/redo_62_no_norm_k1/train_df.csv'), index=False)
redone_test_df.to_csv(os.path.expanduser('~/ryandew_adaptive_preference/RA/results_analysis/redo_62_no_norm_k1/test_df.csv'), index=False)

test_df = redone_test_df


## Set Responses Count
num_responses = test_df.shape[0]
experiment_type = 'mpc'



## Convert String to Arrays and Dictionaries

for column in train_df.columns[9:]:
    try:
        train_df[column] = train_df[column].apply(eval)
    except:
        try:
            train_df[column] = [eval(train_df[column][i]) for i in range(train_df.shape[0])]
        except:
            print(f"exception in normalizing data type for column {column} in training data")
            pass

for column in test_df.columns[13:]:
    try:
        test_df[column] = test_df[column].apply(eval)
    except:
        try:
            test_df[column] = [eval(test_df[column][i]) for i in range(test_df.shape[0])]
        except:
            print(f"exception in normalizing data type for column {column} in test data")
            pass


## Create Access Arrays for Redisplay Check Train Ratings
## All _model_ratings and _user_ratings arrays are 2D indexed by user and then rating (by model or user)
## All _differences arrays are 1D of all ratings across every user

def create_access_arrays_by_index(df, col):
    access_model_ratings = []
    access_user_ratings = []
    access_differences = []
    for i in range(0, num_responses):
        model_ratings = []
        user_ratings = []

        # Here we are just looping over the rows by index, hence the fn name
        for k in range(0, len(df[col][i])):
            model_rating = df[col][i][k][0]
            user_rating = df[col][i][k][1]
            error = model_rating - user_rating

            model_ratings.append(model_rating)
            user_ratings.append(user_rating)
            access_differences.append(error)

        access_model_ratings.append(model_ratings)
        access_user_ratings.append(user_ratings)

    return access_model_ratings, access_user_ratings, access_differences

redisplay_train_model_ratings, redisplay_train_user_ratings, redisplay_train_differences = create_access_arrays_by_index(train_df, 'redisplay_check')



## Create access arrays for attention checks

redisplay_test_model_ratings, redisplay_test_user_ratings, redisplay_test_differences = create_access_arrays_by_index(test_df, 'redisplay_check')
arbitrary_train_model_ratings, arbitrary_train_user_ratings, arbitrary_train_differences = create_access_arrays_by_index(train_df, 'arbitrary_check')
arbitrary_test_model_ratings, arbitrary_test_user_ratings, arbitrary_test_differences = create_access_arrays_by_index(test_df, 'arbitrary_check')



## Convert to Numpy Arrays for Array-Level Functions

redisplay_train_model_ratings = np.array(redisplay_train_model_ratings)
redisplay_train_user_ratings = np.array(redisplay_train_user_ratings)
redisplay_train_differences = np.array(redisplay_train_differences)

redisplay_test_model_ratings = np.array(redisplay_test_model_ratings)
redisplay_test_user_ratings = np.array(redisplay_test_user_ratings)
redisplay_test_differences = np.array(redisplay_test_differences)

arbitrary_train_model_ratings = np.array(arbitrary_train_model_ratings)
arbitrary_train_user_ratings = np.array(arbitrary_train_user_ratings)
arbitrary_train_differences = np.array(arbitrary_train_differences)

arbitrary_test_model_ratings = np.array(arbitrary_test_model_ratings)
arbitrary_test_user_ratings = np.array(arbitrary_test_user_ratings)
arbitrary_test_differences = np.array(arbitrary_test_differences)



## Create Binary Arrays for Every Redisplay Check (1 = Pass, 0 = Fail)

def create_redisplay_hit_vector(differences):
    count = len(differences)
    hit_vector = [0] * count
    for i in range(0, count):
        if (abs(differences[i]) < 0.2):
            hit_vector[i] = 1
    return hit_vector

redisplay_train_hit_vector = create_redisplay_hit_vector(redisplay_train_differences)
redisplay_test_hit_vector = create_redisplay_hit_vector(redisplay_test_differences)



## Create Binary Arrays for Every Arbitrary Check (1 = Pass, 0 = Fail)

def create_arbitrary_hit_vector(differences):
    count = len(differences)
    hit_vector = [0] * count
    for i in range(0, count):
        if (abs(differences[i]) == 0):
            hit_vector[i] = 1
    return hit_vector

arbitrary_train_hit_vector = create_arbitrary_hit_vector(arbitrary_train_differences)
arbitrary_test_hit_vector = create_arbitrary_hit_vector(arbitrary_test_differences)


## Summarize performance on attention checks

redisplay_train_count = len(redisplay_train_differences)
redisplay_test_count = len(redisplay_test_differences)



## Create 1D Binary Array to Indicate if User Passed All Redisplay Checks

passed_redisplay_checks = [1] * num_responses

redisplay_train_individual_diffs = redisplay_train_model_ratings - redisplay_train_user_ratings
redisplay_test_individual_diffs = redisplay_test_model_ratings - redisplay_test_user_ratings

for i in range(0, num_responses):
    for diff in redisplay_train_individual_diffs[i]:
        # For MPC, any two adjacent choices is also considered fine.
        # The difference between them is 0.5 then.
        # For AR, this value 
        if (abs(diff) > 0.51): 
            passed_redisplay_checks[i] = 0
    for diff in redisplay_test_individual_diffs[i]:
        if (abs(diff) > 0.51):
            passed_redisplay_checks[i] = 0

passed_redisplay_checks = np.array(passed_redisplay_checks)


## Create 1D Binary Array to Indicate if User Passed All Arbitrary Checks

passed_arbitrary_checks = [1] * num_responses

arbitrary_train_individual_diffs = arbitrary_train_model_ratings - arbitrary_train_user_ratings
arbitrary_test_individual_diffs = arbitrary_test_model_ratings - arbitrary_test_user_ratings

for i in range(0, num_responses):
    for diff in arbitrary_train_individual_diffs[i]:
        if (diff != 0):
            passed_arbitrary_checks[i] = 0
    for diff in arbitrary_test_individual_diffs[i]:
        if (diff != 0):
            passed_arbitrary_checks[i] = 0

passed_arbitrary_checks = np.array(passed_arbitrary_checks)






## Create Access Arrays for Train Ratings
## All _model_ratings and _user_ratings arrays are 2D indexed by user and then rating (by model or user)
## All _differences arrays are 1D of all ratings across every user

def create_access_arrays_by_key(df, col):
    access_model_ratings = []
    access_user_ratings = []
    access_differences = []
    for i in range(0, num_responses):
        model_ratings = []
        user_ratings = []

        # Loop over keys in the cell of the column
        for k in df[col][i]:
            model_rating = df[col][i][k][0]
            user_rating = df[col][i][k][1]
            error = model_rating - user_rating

            model_ratings.append(model_rating)
            user_ratings.append(user_rating)
            access_differences.append(error)

        access_model_ratings.append(model_ratings)
        access_user_ratings.append(user_ratings)

    return access_model_ratings, access_user_ratings, access_differences

train_model_ratings, train_user_ratings, train_differences = create_access_arrays_by_key(train_df, 'history_dict')



# Test was renamed from "fixed" to "random"
if "fixed_test_predictions_ratings" in test_df.columns:
    random_fixed_test_column_name = "fixed_test_predictions_ratings"
else:
    random_fixed_test_column_name = "random_test_predictions_ratings"

try:    
    test_df[random_fixed_test_column_name] = test_df[random_fixed_test_column_name].apply(eval)
except:
    try:
        test_df[random_fixed_test_column_name] = [eval(test_df[random_fixed_test_column_name][i]) for i in range(test_df.shape[0])]
    except:
        print('error processing random_fixed_test_column formatting')
        pass



## Create Access Arrays for Fixed Test Ratings
fixed_test_model_ratings, fixed_test_user_ratings, fixed_test_differences = create_access_arrays_by_key(test_df, random_fixed_test_column_name)

## Create Access Arrays for Good Test Ratings
good_test_model_ratings, good_test_user_ratings, good_test_differences = create_access_arrays_by_key(test_df, 'good_test_predictions_ratings')



## Convert to Numpy Arrays for Array-Level Functions

train_model_ratings = np.array(train_model_ratings, dtype=object)
train_user_ratings = np.array(train_user_ratings, dtype=object)
train_differences = np.array(train_differences)

fixed_test_model_ratings = np.array(fixed_test_model_ratings)
fixed_test_user_ratings = np.array(fixed_test_user_ratings)
fixed_test_differences = np.array(fixed_test_differences)

good_test_model_ratings = np.array(good_test_model_ratings)
good_test_user_ratings = np.array(good_test_user_ratings)
good_test_differences = np.array(good_test_differences)


def compute_good_accuracy(random_train_user_ratings, good_test_user_ratings):
    user_avg = np.mean(random_train_user_ratings)
    true_good = (good_test_user_ratings > user_avg).sum()
    false_good = (good_test_user_ratings <= user_avg).sum()
    total = sum([true_good, false_good])

    return true_good / total

def compute_mpc_accuracy(model_ratings, user_ratings):
    if experiment_type == 'ratings':
        return np.nan
    
    correct_predictions = [1 if (np.sign([model_ratings[i]]) == np.sign([user_ratings[i]]) 
                                 or user_ratings[i] == 0.0) else 0 for i in range(len(user_ratings))]
    return sum(correct_predictions)/len(correct_predictions)


all_good_test_accuracy = compute_good_accuracy(fixed_test_user_ratings, good_test_user_ratings)



## Construct Individual Level Dataframe w/ Performance Results and Attention Check Pass Indicators

individual_results_df = pd.DataFrame(columns=["user_id", "num_train_ratings", "train_corr", "fixed_test_corr", "good_test_corr",
                                              "train_rmse", "fixed_test_rmse", "good_test_rmse",
                                              "passed_redisplay", "passed_arbitrary", "ratings_accuracy", 
                                              'fixed_mpc_accuracy', 'good_mpc_accuracy'])

for i in range(0, num_responses):
    user_id = train_df.index[i]
    num_train_ratings = len(train_model_ratings[i])
    train_corr, _ = stats.pearsonr(train_model_ratings[i], train_user_ratings[i])
    fixed_test_corr, _ = stats.pearsonr(fixed_test_model_ratings[i], fixed_test_user_ratings[i])
    good_test_corr, _ = stats.pearsonr(good_test_model_ratings[i], good_test_user_ratings[i])

    train_diffs = list(map(operator.sub, train_model_ratings[i], train_user_ratings[i]))
    train_rmse = (np.sum(list(map(operator.mul, train_diffs, train_diffs))) / len(train_diffs)) ** (1 / 2)
    fixed_test_diffs = fixed_test_model_ratings[i] - fixed_test_user_ratings[i]
    fixed_test_rmse = (np.sum(fixed_test_diffs * fixed_test_diffs) / len(fixed_test_diffs)) ** (1 / 2)
    good_test_diffs = good_test_model_ratings[i] - good_test_user_ratings[i]
    good_test_rmse = (np.sum(good_test_diffs * good_test_diffs) / len(good_test_diffs)) ** (1 / 2)

    passed_redisplay = passed_redisplay_checks[i]
    passed_arbitrary = passed_arbitrary_checks[i]

    ratings_accuracy = compute_good_accuracy(train_model_ratings[i], good_test_user_ratings[i])
    fixed_mpc_accuracy = compute_mpc_accuracy(fixed_test_model_ratings[i], fixed_test_user_ratings[i])
    good_mpc_accuracy = compute_mpc_accuracy(good_test_model_ratings[i], good_test_user_ratings[i])

    individual_results_df.loc[len(individual_results_df)] = [user_id, num_train_ratings, train_corr, fixed_test_corr, good_test_corr,
                                                             train_rmse, fixed_test_rmse, good_test_rmse,
                                                             passed_redisplay, passed_arbitrary, ratings_accuracy,
                                                             fixed_mpc_accuracy, good_mpc_accuracy]


# Save individual results to files

individual_results_df.to_csv(os.path.expanduser('~/ryandew_adaptive_preference/RA/results_analysis/redo_62_no_norm_k1/individuals.csv'), index=False)
