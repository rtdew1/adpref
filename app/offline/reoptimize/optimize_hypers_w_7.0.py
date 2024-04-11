import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange
from scipy.optimize import minimize

os.environ['ADAPTIVE_PREFERENCE_ENV'] = 'dev'

# Step 1: Load the Z data
os.chdir(os.path.expanduser("~/ryandew_adaptive_preference/RA/backend/"))
from data_io.mpc.dataset import z_all

# Step 2: Import the PrefOptim object:
os.chdir(os.path.expanduser("~/ryandew_adaptive_preference/RA/results_analysis/reoptimize_mpc/"))
from og_reopt import PrefOptim

# Step 3: Import useful analysis functions:
os.chdir(os.path.expanduser("~/ryandew_adaptive_preference/RA/results_analysis/reoptimize_mpc/"))
from utils import *


## BEGIN ANALYSIS ------------------------------

os.chdir(os.path.expanduser("~/ryandew_adaptive_preference/RA/results_analysis/reoptimize_mpc/"))
og_train = pd.read_csv("inputs/7.0_knn_binary_train_df.csv")
og_test = pd.read_csv("inputs/7.0_knn_binary_test_df.csv")


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

# Refit people using the symmetric GP model, test different inits

def refit_person(i):
    # Symmetric training
    sym_z, sym_ratings = extract_sym(i)
    # Extract test
    test_items, test_ratings, old_preds = extract_test(i)

    # Make predictions under new model + sym settings
    new_prefs = PrefOptim(z_shown = sym_z, ratings = sym_ratings)
    new_prefs.hypers = new_hypers
    sym_preds_new = new_prefs.compute_utility(z_all[test_items])
    
    fit_i = fit_comparison_dict(test_ratings, old_preds, sym_preds_new)
    fit_i.update(hyper_comparison_dict(orig_prefs, new_prefs))

    return fit_i

for i in trange(og_train.shape[0]):   
    fit_data_i = refit_person(i)
    if i == 0:
        fit_data = [fit_data_i]
    else:
        fit_data.append(fit_data_i)

fit_data = pd.DataFrame.from_records(fit_data)

print("fraction strictly better binary acc:", (fit_data['new_sign_acc'] > fit_data['old_sign_acc']).mean())
print("fraction better or equal binary acc:", (fit_data['new_sign_acc'] >= fit_data['old_sign_acc']).mean())
print("fraction better cor:", (fit_data['new_cor'] >= fit_data['old_cor']).mean())
print("fraction better rmse:", (fit_data['new_rmse'] < fit_data['old_rmse']).mean())
print("new mean cor: ", fit_data['new_cor'].mean())
print("old mean cor: ", fit_data['old_cor'].mean())
print("new mean rmse: ", fit_data['new_rmse'].mean())
print("old mean rmse: ", fit_data['old_rmse'].mean())



# Reoptimize the hyperparameters using test RMSE as loss

def rmse_under_hypers(hypers_list, prefs, test_z, test_ratings):
    prefs.hypers = {'noise': hypers_list[0], 'amp': hypers_list[1], 'ls': hypers_list[2]}
    pred = prefs.compute_utility(test_z)
    return rmse(pred, test_ratings)

def reoptim_hypers(pars0, prefs, test_z, test_ratings):
    bnds = ((1e-6, 100), (1e-6, 100), (1e-6, 100))
    opt_out = minimize(rmse_under_hypers, pars0, args=(prefs, test_z, test_ratings), method="L-BFGS-B", bounds=bnds)
    return {"noise": opt_out.x[0], "amp": opt_out.x[1], "ls": opt_out.x[2]}

for i in trange(og_train.shape[0]):
    z_shown, ratings = extract_train(i)
    z_sym, ratings_sym = extract_sym(i)
    test_items, test_ratings, old_preds = extract_test(i)

    # Make predictions under original settings
    orig_prefs = PrefOptim(z_shown = z_sym, ratings = ratings_sym)
    sym_preds_orig = orig_prefs.compute_utility(z_all[test_items])
    orig_rmse = rmse(test_ratings, sym_preds_orig)

    # Make predictions under new settings
    new_prefs = PrefOptim(z_shown = z_sym, ratings = ratings_sym)
    reoptimed_hypers = reoptim_hypers(
        pars0=[0.1,1,0.1], 
        prefs=new_prefs, 
        test_z=z_all[test_items], 
        test_ratings=test_ratings
    )
    new_prefs.hypers = reoptimed_hypers
    sym_preds_new = new_prefs.compute_utility(z_all[test_items])
    new_rmse = rmse(test_ratings, sym_preds_new)
    
    # Save results in a list
    if i == 0:
        orig_hypers_list = [orig_prefs.hypers]
        reoptimed_hypers_list = [reoptimed_hypers]
        rmse_list = [[orig_rmse, new_rmse]]
        preds_list = [[sym_preds_orig, sym_preds_new]]
    else:
        orig_hypers_list.append(orig_prefs.hypers)
        reoptimed_hypers_list.append(reoptimed_hypers)
        rmse_list.append([orig_rmse, new_rmse])
        preds_list.append([sym_preds_orig, sym_preds_new])
    

reoptim_res_df = pd.DataFrame(reoptimed_hypers_list)
reoptim_res_df['old_noise'] = [a['noise'] for a in orig_hypers_list]
reoptim_res_df['old_amp'] = [a['amp'] for a in orig_hypers_list]
reoptim_res_df['old_ls'] = [a['ls'] for a in orig_hypers_list]
reoptim_res_df['old_rmse'] = [a[0] for a in rmse_list]
reoptim_res_df['new_rmse'] = [a[1] for a in rmse_list]
reoptim_res_df['old_preds'] = [a[0] for a in preds_list]
reoptim_res_df['new_preds'] = [a[1] for a in preds_list]

plt.scatter(reoptim_res_df['old_rmse'], reoptim_res_df['new_rmse'])
# Add a line x=y
x = np.linspace(0.3, 0.65, 100)
plt.plot(x, x, color='red')
plt.xlabel("Old RMSE")
plt.ylabel("New RMSE")
plt.title("Old vs. New RMSE")
plt.show()

# Sort reoptim_res_df by difference between old_rmse and new_rmse
reoptim_res_df['rmse_diff'] = reoptim_res_df['new_rmse'] - reoptim_res_df['old_rmse']
reoptim_res_df = reoptim_res_df.sort_values(by='rmse_diff', ascending=True)
reoptim_res_df

# Plot the old and new predictions for the person with the largest rmse_diff
i = reoptim_res_df.index[0]
plt.scatter(reoptim_res_df['old_preds'][i], reoptim_res_df['new_preds'][i])
# Add a line x=y
x = np.linspace(-0.5, 0.5, 100)
plt.plot(x, x, color='red')
plt.xlabel("Old Predictions")
plt.ylabel("New Predictions")
plt.title("Old vs. New Predictions")
plt.show()

# Plot the difference between old and new ls, amp, and noise versus rmse_diff in 3 subplots
fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
axs[0].scatter(reoptim_res_df['ls'] - reoptim_res_df['old_ls'], reoptim_res_df['rmse_diff'])
axs[0].set_xlabel("Difference between old and new ls")
axs[0].set_ylabel("Difference between old and new RMSE")
axs[0].set_title("Difference between old and new ls vs. RMSE")
axs[1].scatter(reoptim_res_df['amp'] - reoptim_res_df['old_amp'], reoptim_res_df['rmse_diff'])
axs[1].set_xlabel("Difference between old and new amp")
axs[1].set_ylabel("Difference between old and new RMSE")
axs[1].set_title("Difference between old and new amp vs. RMSE")
axs[2].scatter(reoptim_res_df['noise'] - reoptim_res_df['old_noise'], reoptim_res_df['rmse_diff'])
axs[2].set_xlabel("Difference between old and new noise")
axs[2].set_ylabel("Difference between old and new RMSE")
axs[2].set_title("Difference between old and new noise vs. RMSE")
plt.show()

# Plot the difference between old and new ls versus rmse_diff
plt.scatter(reoptim_res_df['ls'] - reoptim_res_df['old_ls'], reoptim_res_df['rmse_diff'])
plt.xlabel("Difference between old and new ls")
plt.ylabel("Difference between old and new RMSE")
plt.title("Difference between old and new ls vs. RMSE")
plt.show()

# Plot the difference between old and new amp versus rmse_diff
plt.scatter(reoptim_res_df['amp'] - reoptim_res_df['old_amp'], reoptim_res_df['rmse_diff'])
plt.xlabel("Difference between old and new amp")
plt.ylabel("Difference between old and new RMSE")
plt.title("Difference between old and new amp vs. RMSE")
plt.show()

# Plot the difference between old and new noise versus rmse_diff
plt.scatter(reoptim_res_df['noise'] - reoptim_res_df['old_noise'], reoptim_res_df['rmse_diff'])
plt.xlabel("Difference between old and new noise")
plt.ylabel("Difference between old and new RMSE")
plt.title("Difference between old and new noise vs. RMSE")
plt.show()

# Plot the distribution of all ls, amp, and noise using subplots
def plot_hyper_dists(res_df):
    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
    axs[0].hist(res_df['ls'], bins=20)
    axs[0].set_xlabel("ls")
    axs[0].set_ylabel("Frequency")
    axs[0].set_title("Distribution of ls")
    axs[1].hist(res_df['amp'], bins=20)
    axs[1].set_xlabel("amp")
    axs[1].set_ylabel("Frequency")
    axs[1].set_title("Distribution of amp")
    axs[2].hist(res_df['noise'], bins=20)
    axs[2].set_xlabel("noise")
    axs[2].set_ylabel("Frequency")
    axs[2].set_title("Distribution of noise")
    plt.show()

plot_hyper_dists(reoptim_res_df)
plot_hyper_dists(reoptim_res_df[reoptim_res_df['rmse_diff'] < 0])
plot_hyper_dists(reoptim_res_df.iloc[0:50])

reoptim_res_df[reoptim_res_df['rmse_diff'] < 0].describe()

# Look at predictions for people with noise greater than 1
noise_gt1 = reoptim_res_df[reoptim_res_df['noise'] > 1]
# Plot the old and new predictions for the person with the largest rmse_diff
i = noise_gt1.index[4]
plt.scatter(noise_gt1['old_preds'][i], noise_gt1['new_preds'][i])
# Add a line x=y
x = np.linspace(-0.5, 0.5, 100)
plt.plot(x, x, color='red')
plt.xlabel("Old Predictions")
plt.ylabel("New Predictions")
plt.title("Old vs. New Predictions")
plt.show()

# Plot pairwise scatterplots for noise, amp, and ls
import seaborn as sns
sns.pairplot(reoptim_res_df[['noise', 'amp', 'ls']])


# Reoptimize the hyperparameters using new priors

for i in trange(og_train.shape[0]):
    z_shown, ratings = extract_train(i)
    z_sym, ratings_sym = extract_sym(i)
    test_items, test_ratings, old_preds = extract_test(i)

    # Make predictions under original settings
    orig_prefs = PrefOptim(z_shown = z_sym, ratings = ratings_sym)
    sym_preds_orig = orig_prefs.compute_utility(z_all[test_items])
    orig_rmse = rmse(test_ratings, sym_preds_orig)

    # Make predictions under new settings
    new_prefs = PrefOptim(
        z_shown = z_sym, 
        ratings = ratings_sym, 
        init = [0.1, 1, 0.1],
        addl_optim_args = {
            "noise_prior": {"a": 5, "scale": 20},
            #"noise_prior": {"a": 1, "scale": 0.5},
            "amp_prior": {"a": 2, "scale": 0.5},
            "ls_prior": {"a": 2, "scale": 2}
        }
    )
    new_hypers = new_prefs.hypers
    sym_preds_new = new_prefs.compute_utility(z_all[test_items])

    fit_data_i = fit_comparison_dict(test_ratings, sym_preds_orig, sym_preds_new)
    fit_data_i.update(hyper_comparison_dict(orig_prefs, new_prefs))
    
    # Save results in a list
    if i == 0:
        combined_data = [fit_data_i]
    else:
        combined_data.append(fit_data_i)

combined_data = pd.DataFrame.from_records(combined_data)

# How many times is new_sign_acc higher than old_sign_acc?
print("fraction strictly better binary acc:", (combined_data['new_sign_acc'] > combined_data['old_sign_acc']).mean())
print("fraction better or equal binary acc:", (combined_data['new_sign_acc'] >= combined_data['old_sign_acc']).mean())
print("fraction better cor:", (combined_data['new_cor'] >= combined_data['old_cor']).mean())
print("fraction better rmse:", (combined_data['new_rmse'] < combined_data['old_rmse']).mean())
print("new mean cor: ", combined_data['new_cor'].mean())
print("old mean cor: ", combined_data['old_cor'].mean())
print("new mean rmse: ", combined_data['new_rmse'].mean())
print("old mean rmse: ", combined_data['old_rmse'].mean())

