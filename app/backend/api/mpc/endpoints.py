from api.shared.util import create_user, first_element_not_in
from api.shared.attention_check import (
    handle_attention_check_receive,
    handle_attention_check_send,
    get_redisplay_check_results,
    get_arbitrary_check_results,
)
from api.mpc.responses import move_on, show_image, do_arbitrary_check
from api.mpc.util import rescale_ratings
from api.shared.responses import completed, reply_mturk_id
from data_io.mpc.dataset import all_items, z_all
from data_io.shared.user import get_user_config, get_user_data, save_user_data, update_user_finished
from data_io.shared.output import save_learning_history, save_test_history
from config import UPDATE_HYPERS_ITERS, USE_MAP, APP_VERSION, RESCALE_NORMALIZE
from model import PrefOptim
import numpy as np
import time
import uuid
from flask import request


def post_create_user():
    user_id = create_user(all_items.shape[0])
    return {"user_id": user_id}


def get_initial_send(user_id: uuid.UUID):
    user_id = str(user_id)

    user_config = get_user_config(user_id)
    user_data = get_user_data(user_id)

    num_done_ratings = len(user_data["initial_ratings"])

    if num_done_ratings == 0:
        user_data["time_initial_start"] = time.time()

    if num_done_ratings == user_config["NUM_INITIAL_RATINGS"]:
        user_data["time_initial_end"] = time.time()
        save_user_data(user_id, user_data)
        return move_on()

    num_displayed_items = len(user_data["initial_items_shown"])

    if num_displayed_items > num_done_ratings:  # Duplicated request, last item has not been rated
        item_id = user_data["initial_items_shown"][-1]
        return show_image(item_id)
    else:
        next_item_ix = num_done_ratings  # 0-th item should be shown when 0 ratings are done, etc.
        item_id = user_data["initial_items"][next_item_ix]

        user_data["initial_items_shown"].append(item_id)
        save_user_data(user_id, user_data)
        return show_image(item_id)


def post_initial_receive(user_id: uuid.UUID):
    user_id = str(user_id)

    user_config = get_user_config(user_id)
    user_data = get_user_data(user_id)
    num_done_ratings = len(user_data["initial_ratings"])

    if num_done_ratings < user_config["NUM_INITIAL_RATINGS"]:  # Only handle the request when the last item is not rated
        rating = rescale_ratings(float(request.form["rating"]))
        user_data["initial_ratings"].append(rating)

    save_user_data(user_id, user_data)
    return completed()


def post_build_model(user_id: uuid.UUID):
    user_id = str(user_id)
    user_data = get_user_data(user_id)

    z_shown = z_all[user_data["initial_items_shown"]]
    ratings = user_data["initial_ratings"]
    user_model = PrefOptim(np.array(z_shown), np.array(ratings))

    user_data["pref_model"] = user_model
    save_user_data(user_id, user_data)

    return completed()


def get_refine_send(user_id: uuid.UUID):
    user_id = str(user_id)

    user_config = get_user_config(user_id)
    user_data = get_user_data(user_id)

    num_done_ratings = len(user_data["refine_ratings"])

    if num_done_ratings == 0:
        user_data["time_refine_start"] = time.time()

    if num_done_ratings == user_config["NUM_REFINE_RATINGS"]:
        user_data["time_refine_end"] = time.time()
        save_user_data(user_id, user_data)
        return move_on()

    result, redisplay_item_id, arbitrary_instruction = handle_attention_check_send("refine", user_data)
    if result == "arbitrary":
        return do_arbitrary_check(arbitrary_instruction)

    elif result == "redisplay":
        return show_image(redisplay_item_id)

    # If none of the previous conditions apply, it's a normal step during the refine stage
    num_displayed_items = len(user_data["refine_items_shown"])

    if num_displayed_items > num_done_ratings:  # An item was displayed but not rated
        item_id = user_data["refine_items_shown"][-1]  # Get the last item displayed
        return show_image(item_id)

    else:  # Fetch the requested item from the model
        sorted_next_items = user_data["pref_model"].next_item(z_all)
        item_id = first_element_not_in(
            sorted_next_items, user_data["initial_items_shown"], user_data["refine_items_shown"]
        )
        user_data["refine_items_shown"].append(item_id)
        save_user_data(user_id, user_data)
        return show_image(item_id)


def post_refine_receive(user_id: uuid.UUID):
    user_id = str(user_id)

    user_config = get_user_config(user_id)
    user_data = get_user_data(user_id)

    rating = rescale_ratings(float(request.form["rating"]))
    result, user_data = handle_attention_check_receive("refine", rating, user_data)

    num_done_ratings = len(user_data["refine_ratings"])
    num_items_shown = len(user_data["refine_items_shown"])

    # If not attention check, and not a duplicated post request
    if result == "none" and num_done_ratings < num_items_shown:
        item_id = user_data["refine_items_shown"][-1]
        user_data["refine_ratings"].append(rating)

        # Regardless of the iteration,
        # create the symmetric data point, that takes negation on both the rating and the representation difference,
        # without updating the hypers
        user_data["pref_model"].update_posterior(-np.array([rating]), -z_all[item_id], update_hypers=False)

        # Update every UPDATE_HYPERS_ITERS items, or at the end of the refine stage
        update_hypers = (
            num_done_ratings % UPDATE_HYPERS_ITERS == 0 or num_done_ratings == user_config["NUM_REFINE_RATINGS"]
        )
        user_data["pref_model"].update_posterior(np.array([rating]), z_all[item_id], update_hypers=update_hypers)
        if num_done_ratings == user_config["NUM_REFINE_RATINGS"]:
            print("Last refine rating!")

        if update_hypers:
            user_data["hyper_history"].append(user_data["pref_model"].hypers)

    save_user_data(user_id, user_data)
    return completed()


def post_calc_results(user_id: uuid.UUID):
    user_id = str(user_id)

    user_config = get_user_config(user_id)
    user_data = get_user_data(user_id)

    if RESCALE_NORMALIZE:
        user_data["pref_model"].ends_training()
    predicted_utility_all = user_data["pref_model"].compute_utility(z_all)

    initial_model_fit = predicted_utility_all[user_data["initial_items_shown"]].tolist()
    refine_model_fit = predicted_utility_all[user_data["refine_items_shown"]].tolist()

    learning_history = {
        "user_id": user_id,
        "app_version": APP_VERSION,
        "model_version": user_data["pref_model"].version,
        "status": user_config["status"],
        "NUM_INITIAL_RATINGS": user_config["NUM_INITIAL_RATINGS"],
        "NUM_REFINE_RATINGS": user_config["NUM_REFINE_RATINGS"],
        "TOTAL_TRAINING_RATINGS": user_config["TOTAL_TRAINING_RATINGS"],
        "NUM_TEST_RATINGS": user_config["NUM_GOOD_RATINGS"] + user_config["NUM_RANDOM_RATINGS"],
        "items_shown": str(user_data["initial_items_shown"] + user_data["refine_items_shown"]),
        "ratings": str(user_data["initial_ratings"] + user_data["refine_ratings"]),
        "model_fit": str(initial_model_fit + refine_model_fit),
        "history_dict": str(
            {
                **{
                    user_data["initial_items_shown"][i]: (initial_model_fit[i], user_data["initial_ratings"][i])
                    for i in range(user_config["NUM_INITIAL_RATINGS"])
                },
                **{
                    user_data["refine_items_shown"][i]: (refine_model_fit[i], user_data["refine_ratings"][i])
                    for i in range(user_config["NUM_REFINE_RATINGS"])
                },
            }
        ),
        "hypers": str(user_data["pref_model"].hypers),
        "redisplay_check": str(get_redisplay_check_results("refine", user_data)),
        "arbitrary_check": str(get_arbitrary_check_results("refine", user_data, rescale_fn=rescale_ratings)),
        "hyper_history": str(user_data["hyper_history"]),
    }

    save_learning_history(user_id, learning_history)

    ordered_items = np.argsort(predicted_utility_all)

    unseen_items_sorted_best_first = np.setdiff1d(
        ordered_items[::-1], user_data["initial_items_shown"] + user_data["refine_items_shown"], assume_unique=True
    ).tolist()

    num_good_items = user_config["NUM_GOOD_RATINGS"]
    test_good_items = unseen_items_sorted_best_first[:num_good_items]
    user_data["test_good_items_prediction"] = {item: predicted_utility_all[item] for item in test_good_items}

    candidate_random_items = np.setdiff1d(unseen_items_sorted_best_first, test_good_items, assume_unique=True).tolist()
    num_random_items = user_config["NUM_RANDOM_RATINGS"]

    test_random_items = np.random.choice(candidate_random_items, num_random_items, replace=False).tolist()
    user_data["test_random_items_prediction"] = {item: predicted_utility_all[item] for item in test_random_items}

    test_items = test_random_items + test_good_items
    np.random.shuffle(test_items)
    user_data["test_items"] = test_items

    save_user_data(user_id, user_data)
    return completed()


def get_test_send(user_id: uuid.UUID):
    user_id = str(user_id)

    user_config = get_user_config(user_id)
    user_data = get_user_data(user_id)

    num_done_ratings = len(user_data["test_ratings"])
    num_test_items = user_config["NUM_GOOD_RATINGS"] + user_config["NUM_RANDOM_RATINGS"]

    if num_done_ratings == 0:
        user_data["time_test_start"] = time.time()

    if num_done_ratings == num_test_items:
        user_data["time_test_end"] = time.time()
        save_user_data(user_id, user_data)
        return move_on()

    result, redisplay_item_id, arbitrary_instruction = handle_attention_check_send("test", user_data)
    if result == "arbitrary":
        return do_arbitrary_check(arbitrary_instruction)

    elif result == "redisplay":
        return show_image(redisplay_item_id)

    num_displayed_items = len(user_data["test_items_shown"])

    if num_displayed_items > num_done_ratings:  # Last image is displayed but not rated, show that again
        item_id = user_data["test_items_shown"][-1]
        return show_image(item_id)

    else:  # Show the next test image
        item_id = user_data["test_items"][num_displayed_items]
        user_data["test_items_shown"].append(item_id)
        save_user_data(user_id, user_data)
        return show_image(item_id)


def post_test_receive(user_id: uuid.UUID):
    user_id = str(user_id)

    user_data = get_user_data(user_id)

    rating = rescale_ratings(float(request.form["rating"]))

    handle_attention_check_receive("test", rating, user_data)
    result, user_data = handle_attention_check_receive("test", rating, user_data)

    num_done_ratings = len(user_data["test_ratings"])
    num_items_shown = len(user_data["test_items_shown"])

    if result == "none" and num_done_ratings < num_items_shown:
        # Only record it if this is not an attention check and the last displayed item is not rated
        user_data["test_ratings"].append(rating)

    save_user_data(user_id, user_data)
    return completed()


def get_load_recs(user_id: uuid.UUID):
    user_id = str(user_id)

    user_config = get_user_config(user_id)
    user_data = get_user_data(user_id)

    test_user_rating_dict = {k: v for k, v in zip(user_data["test_items"], user_data["test_ratings"])}

    test_history = {
        "user_id": user_id,
        "app_version": APP_VERSION,
        "model_version": user_data["pref_model"].version,
        "mturk_id": user_data["mturk_id"],
        "status": "Completed",
        "NUM_INITIAL_RATINGS": user_config["NUM_INITIAL_RATINGS"],
        "NUM_REFINE_RATINGS": user_config["NUM_REFINE_RATINGS"],
        "TOTAL_TRAINING_RATINGS": user_config["TOTAL_TRAINING_RATINGS"],
        "NUM_TEST_RATINGS": user_config["NUM_GOOD_RATINGS"] + user_config["NUM_RANDOM_RATINGS"],
        "TIME_INITIAL": user_data["time_initial_end"] - user_data["time_initial_start"],
        "TIME_REFINE": user_data["time_refine_end"] - user_data["time_refine_start"],
        "TIME_TEST": user_data["time_test_end"] - user_data["time_test_start"],
        "random_test_predictions_ratings": str(
            {
                item: (pred, test_user_rating_dict[item])
                for item, pred in user_data["test_random_items_prediction"].items()
            }
        ),
        "good_test_predictions_ratings": str(
            {
                item: (pred, test_user_rating_dict[item])
                for item, pred in user_data["test_good_items_prediction"].items()
            }
        ),
        "hypers": str(user_data["pref_model"].hypers),
        "redisplay_check": str(get_redisplay_check_results("test", user_data)),
        "arbitrary_check": str(get_arbitrary_check_results("test", user_data, rescale_fn=rescale_ratings)),
    }

    save_test_history(user_id, test_history)
    update_user_finished(user_id, user_data)
    save_user_data(user_id, user_data)
    return reply_mturk_id(user_config)
