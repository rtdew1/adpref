from config import APP_VERSION, PARADIGM
from api.shared.util import get_configuration_set, create_user
from data_io.shared.user import get_user_config, get_user_data, update_user_mturk_id
from flask import request

import uuid


def get_version():
    return {"version": APP_VERSION, "paradigm": PARADIGM}


def get_app_info():
    return {"version": APP_VERSION, "paradigm": PARADIGM}


def post_mturk_receive(user_id: uuid.UUID):
    user_id = str(user_id)
    mturk_id = str(request.form["mturk_id"])

    update_user_mturk_id(user_id, mturk_id)

    return {"completed": True}


def get_num_refining(user_id: uuid.UUID):
    user_config = get_user_config(str(user_id))
    return {"num_refining": user_config["NUM_REFINE_RATINGS"]}


def get_num_testing(user_id: uuid.UUID):
    user_config = get_user_config(str(user_id))
    return {"num_testing": user_config["NUM_GOOD_RATINGS"] + user_config["NUM_RANDOM_RATINGS"]}
