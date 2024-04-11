from typing import List, Tuple, Dict, Any
from config import ENV, PROD_CONFIG_SET, DEV_CONFIG, ATTENTION_CHECK_SETTING
from copy import copy
import random
import numpy as np
import uuid
from data_io.shared.user import save_user_config, save_user_data, create_initial_user_data

# Create and send a fresh user ID to the webpage
def get_configuration_set():
    if ENV == "production":
        settings = copy(random.choice(PROD_CONFIG_SET))

    else:
        settings = copy(DEV_CONFIG)

    # Set TOTAL_TRAINING_RATINGS as the sum of the initial and refine ratings
    settings["TOTAL_TRAINING_RATINGS"] = settings["NUM_INITIAL_RATINGS"] + settings["NUM_REFINE_RATINGS"]
    return settings


def create_user(num_items: int) -> str:
    """
    Create a new user.

    Args:
        num_items: total number of items in the experiment

    Returns:
        the created user's id
    """
    user_id = str(uuid.uuid4())
    configuration = get_configuration_set()

    save_user_config(user_id, configuration)

    all_item_ids = np.arange(num_items)
    initial_items = np.random.choice(all_item_ids, configuration["NUM_INITIAL_RATINGS"], replace=False).tolist()
    user_data = create_initial_user_data(initial_items)

    save_user_data(user_id, user_data)

    return user_id


def first_element_not_in(source: List, *targets: List) -> Any:
    """
    Returns: First element in the source container that is not in any of the target containers.
    """
    for i in source:
        if all([i not in target for target in targets]):
            return i


if __name__ == "__main__":
    # Quick unit test for first_element_not_in
    assert first_element_not_in([2, 3, 4, 5], [2, 10, 11], [6, 3, 2, 5]) == 4
