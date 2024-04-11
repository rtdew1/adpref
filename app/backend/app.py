## Key Contributions by Nikhil Kona and Grant Cho ;)

from flask import Flask
from flask_cors import CORS

from api import (
    get_app_info,
    get_version,
    post_create_user,
    post_mturk_receive,
    get_initial_send,
    post_initial_receive,
    get_num_refining,
    get_num_testing,
    post_build_model,
    get_refine_send,
    post_refine_receive,
    post_calc_results,
    get_test_send,
    post_test_receive,
    get_load_recs,
)

from data_io.shared.user import initialize_db

initialize_db()


# START BACKEND SERVER -----------------------------------------

app = Flask(__name__)
CORS(app)

app.route("/app_info")(get_app_info)
app.route("/version")(get_version)
app.route("/create_user", methods=["POST", "GET"])(post_create_user)  # TODO: Allow only POST in the live version

app.route("/mturk_receive/<uuid:user_id>", methods=["POST"])(post_mturk_receive)

app.route("/num_refining/<uuid:user_id>")(get_num_refining)
app.route("/num_testing/<uuid:user_id>")(get_num_testing)

app.route("/initial_send/<uuid:user_id>")(get_initial_send)
app.route("/initial_receive/<uuid:user_id>", methods=["POST"])(post_initial_receive)

app.route("/build_model/<uuid:user_id>", methods=["POST"])(post_build_model)

app.route("/refine_send/<uuid:user_id>")(get_refine_send)
app.route("/refine_receive/<uuid:user_id>", methods=["POST"])(post_refine_receive)

app.route("/calc_results/<uuid:user_id>", methods=["POST"])(post_calc_results)

app.route("/test_send/<uuid:user_id>")(get_test_send)
app.route("/test_receive/<uuid:user_id>", methods=["POST"])(post_test_receive)

app.route("/load_recs/<uuid:user_id>")(get_load_recs)
