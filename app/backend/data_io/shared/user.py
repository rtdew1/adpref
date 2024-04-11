import sqlite3
from typing import Tuple, Dict, List
from config import DB_PATH, USER_DATA_PATH
import pickle


def initialize_db():
    with sqlite3.connect(DB_PATH, isolation_level=None) as conn:
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS
                        users2(User_ID text PRIMARY KEY,
                                mturk_id text,
                                status text,
                                NUM_INITIAL_RATINGS INT DEFAULT 1,
                                NUM_REFINE_RATINGS INT DEFAULT 10,
                                TOTAL_TRAINING_RATINGS INT,
                                NUM_GOOD_RATINGS INT DEFAULT 5,
                                NUM_RANDOM_RATINGS INT DEFAULT 10,
                                TIME_INITIAL DOUBLE DEFAULT 0.0,
                                TIME_REFINE DOUBLE DEFAULT 0.0,
                                TIME_TEST DOUBLE DEFAULT 0.0);
                                """,
            (),
        )
        conn.commit()


def save_user_config(user_id: str, configuration: Dict) -> None:
    with sqlite3.connect(DB_PATH, isolation_level=None) as conn:
        cur = conn.cursor()
        sql = """INSERT INTO users2(User_ID,
                                    mturk_id,
                                    status,
                                    NUM_INITIAL_RATINGS,
                                    NUM_REFINE_RATINGS,
                                    TOTAL_TRAINING_RATINGS,
                                    NUM_GOOD_RATINGS,
                                    NUM_RANDOM_RATINGS,
                                    TIME_INITIAL,
                                    TIME_REFINE,
                                    TIME_TEST
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"""

        cur.execute(
            sql,
            (
                user_id,
                "N/A",
                "Started",
                configuration["NUM_INITIAL_RATINGS"],
                configuration["NUM_REFINE_RATINGS"],
                configuration["TOTAL_TRAINING_RATINGS"],
                configuration["NUM_GOOD_RATINGS"],
                configuration["NUM_RANDOM_RATINGS"],
                0.0,
                0.0,
                0.0,
            ),
        )
        conn.commit()


def convert_row_to_dict(row: Tuple) -> Dict:
    fields = [
        "User_ID",
        "mturk_id",
        "status",
        "NUM_INITIAL_RATINGS",
        "NUM_REFINE_RATINGS",
        "TOTAL_TRAINING_RATINGS",
        "NUM_GOOD_RATINGS",
        "NUM_RANDOM_RATINGS",
        "TIME_INITIAL",
        "TIME_REFINE",
        "TIME_TEST",
    ]

    return {f: v for f, v in zip(fields, row)}


def get_user_data(user_id: str):
    with open(USER_DATA_PATH + "/" + user_id + ".pickle", "rb") as data_file:
        d = pickle.load(data_file)
        return d


def save_user_data(user_id: str, data: Dict):
    with open(USER_DATA_PATH + "/" + user_id + ".pickle", "wb") as data_file:
        pickle.dump(data, data_file)


def get_user_config(user_id) -> Dict:
    with sqlite3.connect(DB_PATH, isolation_level=None) as conn:
        cur = conn.cursor()
        sql = """SELECT User_ID,
                        mturk_id,
                        status,
                        NUM_INITIAL_RATINGS,
                        NUM_REFINE_RATINGS,
                        TOTAL_TRAINING_RATINGS,
                        NUM_GOOD_RATINGS,
                        NUM_RANDOM_RATINGS,
                        TIME_INITIAL,
                        TIME_REFINE,
                        TIME_TEST FROM users2 WHERE User_ID = ?;"""
        result = cur.execute(sql, (user_id,)).fetchone()
        return convert_row_to_dict(result)


def update_user_mturk_id(user_id: str, mturk_id: str):
    """
    Update user's mturk_id to the new value in both database and pickle files
    """

    # SQL part
    with sqlite3.connect(DB_PATH, isolation_level=None) as conn:
        cur = conn.cursor()
        sql = """UPDATE users2 SET mturk_id = ? WHERE User_ID = ?;"""
        cur.execute(sql, (mturk_id, user_id))
        conn.commit()

    # Pickle part
    user_data = get_user_data(user_id)
    user_data["mturk_id"] = mturk_id
    save_user_data(user_id, user_data)


def update_user_finished(user_id: str, user_data):
    """
    Update user's DB information when user has finished
    """

    with sqlite3.connect(DB_PATH, isolation_level=None) as conn:
        cur = conn.cursor()
        sql = """
        UPDATE users2
        SET
            status = ?,
            TIME_INITIAL = ?,
            TIME_REFINE = ?,
            TIME_TEST = ?
        WHERE User_ID = ?;
        """
        cur.execute(
            sql,
            (
                "Completed",
                user_data["time_initial_end"] - user_data["time_initial_start"],
                user_data["time_refine_end"] - user_data["time_refine_start"],
                user_data["time_test_end"] - user_data["time_test_start"],
                user_id,
            ),
        )
        conn.commit()


def create_initial_user_data(initial_items: List[int]) -> Dict:

    user_data = {
        "initial_items": initial_items,
        "initial_items_shown": [],
        "initial_ratings": [],
        "refine_items_shown": [],
        "refine_ratings": [],
        "pref_model": None,
        "test_items": [],
        "test_ratings": [],
        "test_items_shown": [],
        "test_good_items_prediction": {},
        "test_random_items_prediction": {},
        "time_initial_start": 0.0,
        "time_initial_end": 0.0,
        "time_refine_start": 0.0,
        "time_refine_end": 0.0,
        "time_test_start": 0.0,
        "time_test_end": 0.0,
        "mturk_id": "",
        "refine_redisplay_check_ratings": [],
        "refine_arbitrary_check_ratings": [],
        "test_redisplay_check_ratings": [],
        "test_arbitrary_check_ratings": [],
        "hyper_history": [],
    }

    return user_data
