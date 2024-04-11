from data_io.ar.dataset import all_items


def move_on():
    return {
        "img_url": "",
        "move_on": True,
        "arbitrary_check": -1,
    }


def show_image(item_id: int):
    return {
        "img_url": all_items.loc[item_id, "URL"],
        "move_on": False,
        "arbitrary_check": -1,
    }


def do_arbitrary_check(instruction):
    return {
        "img_url": "",
        "move_on": False,
        "arbitrary_check": instruction,
    }
