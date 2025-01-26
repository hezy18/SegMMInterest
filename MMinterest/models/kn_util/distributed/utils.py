import socket
import torch.distributed as dist
import os
import datetime
import torch
import logging
from functools import wraps
from typing import Any, Callable, Optional, Union

log = logging.getLogger(__name__)


def rank_zero_only(fn: Callable) -> Callable:
    """Function that can be used as a decorator to enable a function/method being called only on global rank 0."""

    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Optional[Any]:
        rank = getattr(rank_zero_only, "rank", None)
        if rank is None:
            raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
        if rank == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


def get_device(model):
    return next(model.parameters()).device


def get_available_port():
    sock = socket.socket()
    sock.bind(('', 0))
    return sock.getsockname()[1]


def get_env(name):
    if name == "local_rank":
        return int(os.getenv("LOCAL_RANK", 0))
    if name == "rank":
        return int(os.getenv("RANK", 0))
    if name == "world_size":
        return int(os.getenv("WORLD_SIZE", 1))


def initialize_ddp_from_env():
    from ..basic import global_get as G
    # local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl",
                            init_method="env://",
                            world_size=world_size,
                            rank=rank,
                            timeout=datetime.timedelta(seconds=5400))

    torch.cuda.set_device(local_rank)
    G("cfg").train.batch_size = G("cfg").train.batch_size // world_size
    print(f"rank#{rank}\t"
          f"env {local_rank}/{rank}/{world_size}\t"
          f"mem {torch.cuda.memory_usage()}")


def is_ddp_initialized_and_available():
    return dist.is_initialized() and dist.is_available()