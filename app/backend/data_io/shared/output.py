from typing import Dict
from config import OUTPUT_DATA_PATH, APP_VERSION
import pandas as pd
import os


def save_csv_output(history_type: str, user_id: str, data_dict: Dict[str, str]) -> None:
    assert history_type in ["learning", "test"]

    app_version_short = APP_VERSION.split("-")[0]

    filename = f"{app_version_short}_{user_id}_{history_type}_history.csv"
    filepath = os.path.join(OUTPUT_DATA_PATH, filename)

    df = pd.DataFrame(data=data_dict, index=[0])
    df.to_csv(filepath, index_label="impression")


def save_learning_history(user_id: str, data_dict: Dict[str, str]) -> None:
    save_csv_output("learning", user_id, data_dict)


def save_test_history(user_id: str, data_dict: Dict[str, str]) -> None:
    save_csv_output("test", user_id, data_dict)
