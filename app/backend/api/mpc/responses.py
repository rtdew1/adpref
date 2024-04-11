from data_io.mpc.dataset import all_items


def move_on():
    return {
        "img_url_1": "",
        "img_url_2": "",
        "move_on": True,
        "arbitrary_check": -1,
    }


def show_image(item_id: int):
    return {
        "img_url_1": all_items.loc[item_id, "URL_1"],
        "img_url_2": all_items.loc[item_id, "URL_2"],
        "move_on": False,
        "arbitrary_check": False,
        "arbitrary_instruction": None,
    }


def do_arbitrary_check(instruction):
    return {
        "img_url_1": "",
        "img_url_2": "",
        "move_on": False,
        "arbitrary_check": True,
        "arbitrary_instruction": instruction,
    }
