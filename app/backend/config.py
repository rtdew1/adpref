import os

# App Version, please remember to increment this every time the app is updated
APP_VERSION = "testing"

# Experiment Paradigm, options: ar, mpc, fe, mpc_binary
PARADIGM = "mpc_binary"

# Normalize amplitude hyperparameter to 1
RESCALE_NORMALIZE = False

# Model configurations, which model to use, options:
# knn, og, ucb, remboish, aggregated, linear, abernethy, gur, gur_sym
MODEL = "aggregated"

UPDATE_HYPERS_ITERS = 1
USE_MAP = True

# Data files & paths
DATA_FILE = "./data/scaled_vgg_pca10_dresses.csv"

ENV = os.environ["ADAPTIVE_PREFERENCE_ENV"]
user_data_path_options = {
    "dev": "../user_data",
    "dev-docker": "/user_data",
    # The naming of "DATA_PATH" is a legacy issue on the production environment. Keeping it as-is for now.
    "production": os.environ["DATA_PATH"] if "DATA_PATH" in os.environ else None,
}
USER_DATA_PATH = user_data_path_options[ENV]
DB_PATH = USER_DATA_PATH + f"/users-v{APP_VERSION}.sqlite3"

output_data_path_options = {
    "dev": "../output_data",
    "dev-docker": "/output_data",
    "production": "s3://edu-upenn-wharton-adaptive-pref/user_results",
}
OUTPUT_DATA_PATH = output_data_path_options[ENV]

# Attention check settings

## Refine stage:

# Redisplay: list of tuple (a, b), show the a-th item again, before user rates the b-th item
refine_redisplay_setting = [(1, 9)]

# Arbitrary: list of tuple (i, s), before user rates the i-th item, instruct them to rate it as s
refine_arbitrary_setting = []


## Test stage, variable meaning same as above:
test_redisplay_setting = [(2, 12)]
test_arbitrary_setting = []

# The users' ratings are aligned with the configurations according to their index in the list
# Sort to make sure that the settings are listed as the order that user sees the check
refine_redisplay_setting = sorted(refine_redisplay_setting, key=lambda x: x[1])
refine_arbitrary_setting = sorted(refine_arbitrary_setting)
test_redisplay_setting = sorted(test_redisplay_setting, key=lambda x: x[1])
test_arbitrary_setting = sorted(test_arbitrary_setting)

ATTENTION_CHECK_SETTING = {
    "refine": {
        "redisplay": refine_redisplay_setting,
        "arbitrary": refine_arbitrary_setting,
    },
    "test": {
        "redisplay": test_redisplay_setting,
        "arbitrary": test_arbitrary_setting,
    },
}


# Number of samples shown during the initial/refine/test stage

# Production environment configuration set, users will be randomly assign one of them
# If using the `fe` paradigm, the NUM_REFINE_RATINGS will be ignored.
# The paradigm's configuration determines how many training items to use in total.
PROD_CONFIG_SET = [
    {
        "NUM_INITIAL_RATINGS": 1,
        "NUM_REFINE_RATINGS": 40,
        "NUM_GOOD_RATINGS": 5,
        "NUM_RANDOM_RATINGS": 10,
    },
]

## Development environment configuration
DEV_CONFIG = PROD_CONFIG_SET[0]
