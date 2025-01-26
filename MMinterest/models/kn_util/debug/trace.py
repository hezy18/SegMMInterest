from loguru import logger
import inspect
import os
import functools


def log_trace(stack_depth=1, print_args=False):

    def log_wrapper(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            stack_trace = inspect.stack()
            RANK = os.getenv('RANK', '0')
            log_str = f"##### STACK TRACE {func.__name__} RANK {RANK}#####"
            max_depth = min(stack_depth, len(stack_trace) - 1)
            for frame_idx in range(1, max_depth + 1):
                frame_info = stack_trace[frame_idx]
                filename = frame_info.filename
                lineno = frame_info.lineno
                func_name = frame_info.function

                frame_args = inspect.getargvalues(frame_info.frame)
                args_str = str(frame_args.locals) if print_args else "**args**"
                log_str += "\n" + f"\t [{frame_idx}] {filename}:{lineno} {func_name}({args_str})"
            logger.info(log_str)
            logger.info(f"##### END STACK TRACE {func.__name__} RANK {RANK}#####")

            return func(*args, **kwargs)

        return wrapper

    return log_wrapper