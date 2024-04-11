from typing import Dict, Tuple, List, Callable
from config import ATTENTION_CHECK_SETTING


def validate_stage(allowed_stages: List[str]):
    def inner1(func):
        def inner2(stage, *args, **kwargs):

            if stage not in allowed_stages:
                raise ValueError(f"Invalid stage: {stage}")
            return func(stage, *args, **kwargs)

        return inner2

    return inner1


@validate_stage(["refine", "test"])
def handle_attention_check_send(stage: str, user_data: Dict) -> Tuple[str, int, int]:
    """
    Handles the attention check sending.
    Args:
        stage: either "refine" or "test"

    Returns: Tuple of three elements
        - result:   the result of this attention check handling, three possible values,
                    - "redisplay": this is a redisplay check;
                    - "arbitrary": this is an arbitrary check;
                    - "none": this is not an attention check.
                    The other return values only have meaning when this is a corresponding type of attention check

        - redisplay_item_id: the item_id that should be re-displayed
        - arbitrary_instruction: the arbitrary rating instruction
    """

    # Unpack the settings and rounds
    redisplay_setting = ATTENTION_CHECK_SETTING[stage]["redisplay"]
    redisplay_items = [t[1] for t in redisplay_setting]  # before these items, do a redisplay check

    arbitrary_setting = ATTENTION_CHECK_SETTING[stage]["arbitrary"]
    arbitrary_items = [t[0] for t in arbitrary_setting]  # before these items, do an arbitrary check

    todo_item = len(user_data[f"{stage}_ratings"])
    # The index of the todo item, assuming that this is not an attention check

    num_done_redisplay = len(user_data[f"{stage}_redisplay_check_ratings"])
    num_done_arbitrary = len(user_data[f"{stage}_arbitrary_check_ratings"])

    if todo_item in redisplay_items:
        redisplay_item_ix = redisplay_items.index(todo_item)  # position of the todo_item in the redisplay settings
        if num_done_redisplay <= redisplay_item_ix:
            # Needs to do redisplay check, e.g. 1 check is done (the 0-th), but the todo item is the 1st item in the setting
            show_item_ix = redisplay_setting[redisplay_item_ix][0]  # Extract the redisplay item from the configuration
            redisplay_item_id = user_data[f"{stage}_items_shown"][show_item_ix]
            return "redisplay", redisplay_item_id, -1

    if todo_item in arbitrary_items:  # Similar to redisplay logic. Please refer to the above explanation
        arbitrary_item_ix = arbitrary_items.index(todo_item)
        if num_done_arbitrary <= arbitrary_item_ix:
            arbitrary_check = arbitrary_setting[arbitrary_item_ix][1]
            return "arbitrary", 0, arbitrary_check

    return "none", 0, -1


@validate_stage(["refine", "test"])
def handle_attention_check_receive(stage: str, rating: float, user_data: Dict) -> Tuple[str, Dict]:
    """
    Handles the attention check receiving.
    Args:
        stage: either "refine" or "test"
        user_data: the user_data dictionary

    Returns: Tuple of two elements
        - result:   the result of this attention check handling, three possible values,
                    - "redisplay": this is a redisplay check;
                    - "arbitrary": this is an arbitrary check;
                    - "none": this is not an attention check.

        - new_user_data: the new user data dictionary with the attention check response saved.
                         If this is not an attention check, the original user_data will be returned
    """

    redisplay_setting = ATTENTION_CHECK_SETTING[stage]["redisplay"]
    redisplay_items = [t[1] for t in redisplay_setting]  # before these items, do a redisplay check

    arbitrary_setting = ATTENTION_CHECK_SETTING[stage]["arbitrary"]
    arbitrary_items = [t[0] for t in arbitrary_setting]  # before these items, do an arbitrary check

    response_item = len(
        user_data[f"{stage}_ratings"]
    )  # The index of the response item, assuming that this is not an attention check

    num_done_redisplay = len(user_data[f"{stage}_redisplay_check_ratings"])
    num_done_arbitrary = len(user_data[f"{stage}_arbitrary_check_ratings"])

    if response_item in redisplay_items:
        redisplay_item_ix = redisplay_items.index(response_item)
        if num_done_redisplay <= redisplay_item_ix:
            user_data[f"{stage}_redisplay_check_ratings"].append(rating)
            return "redisplay", user_data

    if response_item in arbitrary_items:
        arbitrary_item_ix = arbitrary_items.index(response_item)
        if num_done_arbitrary <= arbitrary_item_ix:
            user_data[f"{stage}_arbitrary_check_ratings"].append(rating)
            return "arbitrary", user_data

    return "none", user_data


@validate_stage(["refine", "test"])
def get_redisplay_check_results(stage: str, user_data: Dict) -> List[Tuple]:
    """
    Args:
        stage: either "refine" or "test"
        user_data: the user data dictionary

    Returns:
        List of tuples representing the results of redisplay attention check questions in the specified phase.
        Each tuple is the result of one check question.
        The first element is user's rating when the image is re-displayed.
        The second element is user's rating when the image is displayed for the first time
    """
    setting = ATTENTION_CHECK_SETTING[stage]["redisplay"]
    user_ratings = user_data[f"{stage}_redisplay_check_ratings"]
    return [
        (
            user_rating,
            user_data[f"{stage}_ratings"][a],
        )
        for user_rating, (a, _) in zip(user_ratings, setting)
    ]


@validate_stage(["refine", "test"])
def get_arbitrary_check_results(stage: str, user_data: Dict, rescale_fn: Callable) -> List[Tuple]:
    """
    Args:
        stage: either "refine" or "test"
        user_data: the user data dictionary
        scaling_fn: the rescale function used for ratings

    Returns:
        List of tuples representing the results of arbitrary check questions in the specified phase.
        Each tuple is the result of one check question.
        The first element is user's rating.
        The second element is the rescaled score in the instruction.
    """

    setting = ATTENTION_CHECK_SETTING[stage]["arbitrary"]
    user_ratings = user_data[f"{stage}_arbitrary_check_ratings"]
    return [
        (
            user_rating,
            rescale_fn(s),
        )
        for user_rating, (_, s) in zip(user_ratings, setting)
    ]
