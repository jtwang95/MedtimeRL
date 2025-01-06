import itertools, logging, coloredlogs, time, os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import sys


def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


def datetime_suffix():
    return datetime.now().strftime("%y%m%d%H%M%S")


def random_seed():
    t = 1000 * time.time()  # current time in milliseconds
    seed = int(t) % 2**32
    return seed


def set_home_folder_prefix(path_dict):
    nodename = os.uname()[1]
    if nodename in path_dict.keys():
        return path_dict[nodename]
    else:
        return path_dict["default"]


def create_logger(logger_name="this project", logging_level="info"):
    # level = "info", "warn"m "debug", "error"
    level_map = {
        "info": logging.INFO,
        "warn": logging.WARN,
        "debug": logging.DEBUG,
        "error": logging.ERROR,
    }
    level = level_map[logging_level]
    FORMAT = "%(asctime)s;%(name)s;%(levelname)s|%(message)s"
    DATEF = "%H-%M-%S"
    # logging.basicConfig(format=FORMAT, level=logging.INFO)
    LEVEL_STYLES = dict(
        debug=dict(color="magenta"),
        info=dict(color="green"),
        verbose=dict(),
        warning=dict(color="blue"),
        error=dict(color="yellow"),
        critical=dict(color="red", bold=True),
    )
    LEVEL_STYLES_ALT = dict(
        debug=dict(color="magenta"),
        info=dict(color="cyan"),
        verbose=dict(),
        warning=dict(color="blue"),
        error=dict(color="yellow"),
        critical=dict(color="red", bold=True),
    )

    mylogger = logging.getLogger(name=logger_name)
    coloredlogs.install(
        level=level,
        fmt=FORMAT,
        datefmt=DATEF,
        level_styles=LEVEL_STYLES,
        logger=mylogger,
    )
    return mylogger


def no_tqdm(func):
    """Decorator to disable tqdm output within a function."""

    def wrapper(*args, **kwargs):
        # Temporarily redirect stdout to /dev/null
        old_stdout = sys.stdout
        sys.stdout = open("/dev/null", "w")

        try:
            result = func(*args, **kwargs)
        finally:
            # Restore original stdout
            sys.stdout = old_stdout

        return result

    return wrapper
