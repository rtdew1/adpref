# Imports
import os
import argparse
import glob
import time
import operator
import numpy as np
from numpy import array
import pandas as pd
import scipy.stats as stats
import json
import shutil

train_df = pd.read_csv(os.path.expanduser('~/ryandew_adaptive_preference/RA/results_analysis/redo_41_knn1_2/4-1_train_df.csv'))
test_df = pd.read_csv(os.path.expanduser('~/ryandew_adaptive_preference/RA/results_analysis/redo_41_knn1_2/4-1_knn1_test_df.csv'))
target_dir = "redone_41_knn1"
new_name = target_dir
output_folder = target_dir
text_result = {}
experiment_type = 'ratings'

def reset_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

    os.mkdir(dir_name)

reset_dir(target_dir)

num_responses = train_df.shape[0]


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
text_result["All Training Data Mean Redisplay Check Hit Rate"] = sum(redisplay_train_hit_vector) / redisplay_train_count
text_result["All Test Data Mean Redisplay Check Hit Rate"] = sum(redisplay_test_hit_vector) / redisplay_test_count
text_result["All Training Data Mean Arbitrary Check Hit Rate"] = np.nan # Not doing arbitrary test anymore
text_result["All Test Data Mean Arbitrary Check Hit Rate"] = np.nan


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



text_result["Individual All Redisplay Check Pass Rate"] = sum(passed_redisplay_checks) / num_responses
text_result["Individual All Arbitrary Check Pass Rate"] = sum(passed_arbitrary_checks) / num_responses
both_checks_pass_rate = sum(np.round((passed_redisplay_checks + passed_arbitrary_checks) / 2)) / num_responses
text_result["Individual All of Both Checks Pass Rate"] = both_checks_pass_rate



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
text_result["Good Test Accuracy"] = all_good_test_accuracy



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

train_path = os.path.join(target_dir, "train_df.csv")
test_path = os.path.join(target_dir, "test_df.csv")
individual_path = os.path.join(target_dir, "individuals.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
individual_results_df.to_csv(individual_path, index=False)


# Create summary json file

## Find Earliest Result Date as Study Run Date

earliest_time = 999999


## Gather App/Model Version and Date for Batch Name

app_version = new_name
model_version = new_name
month = time.ctime(earliest_time)[4:7]
day = time.ctime(earliest_time)[8:10]
year = time.ctime(earliest_time)[20:]
date = month + "_" + day + "_" + year


## Fill in various fields...

batch_name = str(app_version) + "_" + str(model_version) + "_" + date
text_result["Batch Name"] = batch_name

all_train_corr = individual_results_df['train_corr'].mean()
text_result["All Training Data Correlation"] = all_train_corr

all_fixed_test_corr = individual_results_df['fixed_test_corr'].mean()
text_result["All Fixed Test Data Correlation"] = all_fixed_test_corr

all_good_test_corr = individual_results_df['good_test_corr'].mean()
text_result["All Good Test Data Correlation"] = all_good_test_corr

all_train_rmse = individual_results_df['train_rmse'].mean()
text_result["All Training Data RMSE"] = all_train_rmse

all_fixed_test_rmse = individual_results_df['fixed_test_rmse'].mean()
text_result["All Fixed Test Data RMSE"] = all_fixed_test_rmse

all_good_test_rmse = individual_results_df['good_test_rmse'].mean()
text_result["All Good Test Data RMSE"] = all_good_test_rmse

ratings_accuracy = individual_results_df['ratings_accuracy'].mean()
text_result["Good Test Ratings-style Accuracy"] = ratings_accuracy

fixed_mpc_accuracy = individual_results_df['fixed_mpc_accuracy'].mean()
text_result["Fixed Test MPC-style Accuracy"] = fixed_mpc_accuracy

good_mpc_accuracy = individual_results_df['good_mpc_accuracy'].mean()
text_result["Good Test MPC-style Accuracy"] = good_mpc_accuracy


with open(os.path.join(target_dir, f"textual_results.json"), "w", encoding="utf-8") as f:
    json.dump(text_result, f, indent=4, separators=(',', ': '))


# Compute all results by number of ratings:

unique_training_ratings = individual_results_df["num_train_ratings"].unique()

for n_ratings in unique_training_ratings:

    indiv_df_n_ratings = individual_results_df[individual_results_df["num_train_ratings"] == n_ratings]
    num_responses_all = indiv_df_n_ratings.shape[0]
    num_responses_passed_r = indiv_df_n_ratings[indiv_df_n_ratings["passed_redisplay"] == 1].shape[0]


    ## Compute averages of individual-level test statistics:

    all_train_corr = indiv_df_n_ratings['train_corr'].mean()
    all_fixed_test_corr = indiv_df_n_ratings['fixed_test_corr'].mean()
    all_good_test_corr = indiv_df_n_ratings['good_test_corr'].mean()
    all_train_rmse = indiv_df_n_ratings['train_rmse'].mean()
    all_fixed_test_rmse = indiv_df_n_ratings['fixed_test_rmse'].mean()
    all_good_test_rmse = indiv_df_n_ratings['good_test_rmse'].mean()
    ratings_accuracy = indiv_df_n_ratings['ratings_accuracy'].mean()
    fixed_mpc_accuracy = indiv_df_n_ratings['fixed_mpc_accuracy'].mean()
    good_mpc_accuracy = indiv_df_n_ratings['good_mpc_accuracy'].mean()


    ## Compute averages by condition:

    passed_r_train_corr = indiv_df_n_ratings[indiv_df_n_ratings['passed_redisplay'] == 1]['train_corr'].mean()
    passed_r_fixed_test_corr = indiv_df_n_ratings[indiv_df_n_ratings['passed_redisplay'] == 1]['fixed_test_corr'].mean()
    passed_r_good_test_corr = indiv_df_n_ratings[indiv_df_n_ratings['passed_redisplay'] == 1]['good_test_corr'].mean()

    passed_r_train_rmse = indiv_df_n_ratings[indiv_df_n_ratings['passed_redisplay'] == 1]['train_rmse'].mean()
    passed_r_fixed_test_rmse = indiv_df_n_ratings[indiv_df_n_ratings['passed_redisplay'] == 1]['fixed_test_rmse'].mean()
    passed_r_good_test_rmse = indiv_df_n_ratings[indiv_df_n_ratings['passed_redisplay'] == 1]['good_test_rmse'].mean()

    passed_r_ratings_accuracy = indiv_df_n_ratings[indiv_df_n_ratings['passed_redisplay'] == 1]['ratings_accuracy'].mean()
    passed_r_fixed_mpc_accuracy = indiv_df_n_ratings[indiv_df_n_ratings['passed_redisplay'] == 1]['fixed_mpc_accuracy'].mean()
    passed_r_good_mpc_accuracy = indiv_df_n_ratings[indiv_df_n_ratings['passed_redisplay'] == 1]['good_mpc_accuracy'].mean()


    ## Create 1D Access Arrays for Individual Hyperparameters

    if ("Noise" in test_df["hypers"][0]):
        test_noise_hypers = []
        test_amp_hypers = []
        test_ls_hypers = []

        for i in range(0, num_responses):
            test_noise_hypers.append(test_df["hypers"][i]["noise"])
            test_amp_hypers.append(test_df["hypers"][i]["amp"])
            test_ls_hypers.append(test_df["hypers"][i]["ls"])

        indiv_df_n_ratings["noise"] = test_noise_hypers
        indiv_df_n_ratings["amp"] = test_amp_hypers
        indiv_df_n_ratings["ls"] = test_ls_hypers


    ## Load Model Performance File if Exists, Otherwise Create New Dataframe

    performance_path = os.path.join(".", "model_performance_df.csv")
    if (not os.path.isfile(performance_path)):
        model_performance_df = pd.DataFrame(
            columns = [
                "app_version", "model_version", "date", "num_train_ratings", 
                "redisplay_filter", "num_responses", 
                "train_corr", "fixed_test_corr", "good_test_corr",
                "train_rmse", "fixed_test_rmse", "good_test_rmse", 
                "ratings_accuracy", "fixed_mpc_accuracy", "good_mpc_accuracy"
                ]
            )
    else:
        model_performance_df = pd.read_csv(performance_path)

    model_performance_df.loc[len(model_performance_df)] = [
        app_version, model_version, date, n_ratings, 
        0, num_responses_all, 
        all_train_corr, all_fixed_test_corr, all_good_test_corr,
        all_train_rmse, all_fixed_test_rmse, all_good_test_rmse,
        ratings_accuracy, fixed_mpc_accuracy, good_mpc_accuracy
    ]

    model_performance_df.loc[len(model_performance_df)] = [
        app_version, model_version, date, n_ratings, 
        1, num_responses_passed_r, 
        passed_r_train_corr, passed_r_fixed_test_corr, passed_r_good_test_corr,
        passed_r_train_rmse, passed_r_fixed_test_rmse, passed_r_good_test_rmse,
        passed_r_ratings_accuracy, passed_r_fixed_mpc_accuracy, passed_r_good_mpc_accuracy
    ]


    # If running script on same app version/model/responses, delete first results and only keep most recent
    model_performance_df = model_performance_df.drop_duplicates(keep="last")
    model_performance_df.to_csv(performance_path, index=False)