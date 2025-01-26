import logging
import os
from termcolor import colored
import functools
import sys
import torch.distributed as dist

logger_initialized = dict()

@functools.lru_cache()
def get_logger(name="", output_dir=None, to_console=True):
    if name in logger_initialized:
        return logger_initialized[name]
    from ..distributed import rank_zero_only
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if dist.is_initialized() and dist.is_available():
        dist_rank = dist.get_rank()
    else:
        dist_rank = 0

    # create formatter
    fmt = "[%(asctime)s]" + "(%(name)s:%(lineno)d)" + "%(levelname)s" + " %(message)s"
    color_fmt = (
        colored("[%(asctime)s]", "green")
        + colored("(%(name)s:%(lineno)d)", "blue")
        + colored("%(levelname)s", "yellow")
        + " %(message)s"
    )
    # fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    # color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
    #             colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(console_handler)

    # create file handlers
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(output_dir, f"log_rank{dist_rank}.txt"), mode="a"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)
    
    logger_initialized[name] = logger

    return logger
