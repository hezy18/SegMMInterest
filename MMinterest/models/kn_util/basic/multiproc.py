from tqdm import tqdm
import time
from pathos.multiprocessing import Pool

from pathos.helpers import mp
from pathos.threading import ThreadPool
from pathos.pools import ProcessPool
import tqdm
import time


def map_async(iterable, func, num_process=30, desc: object = "", test_flag=False):
    """while test_flag=True, run sequentially"""
    if test_flag:
        ret = [func(x) for x in tqdm(iterable)]
        return ret
    else:
        p = Pool(num_process)
        # ret = []
        # for it in tqdm(iterable, desc=desc):
        #     ret.append(p.apply_async(func, args=(it,)))
        ret = p.map_async(func=func, iterable=iterable)
        total = ret._number_left
        pbar = tqdm(total=total, desc=desc)
        while ret._number_left > 0:
            pbar.n = total - ret._number_left
            pbar.refresh()
            time.sleep(0.1)
        p.close()

        return ret.get()


def map_async_with_tolerance(iterable, func, num_workers=32, level="thread", is_ready=lambda x: x):

    if level == "thread":
        p = ThreadPool(num_workers)
    elif level == "process":
        p = ProcessPool(num_workers)
    p.restart()

    data_queue = mp.Queue()
    for x in iterable:
        data_queue.put(x)

    running = []

    total = data_queue.qsize()
    pbar = tqdm(total=total)

    while not (data_queue.empty() and len(running) == 0):
        if not data_queue.empty():
            cur_item = data_queue.get()
            cur_thread = p.apipe(func, cur_item)
            running.append(cur_thread)

        # update running processes whose state in unready
        new_running = []
        for item in running:
            if not item.ready():
                new_running.append(item)
            elif is_ready(item.get()):
                pbar.n = pbar.n + 1
                pbar.refresh()
                time.sleep(0.1)
        running.clear()
        del running
        running = new_running

    p.close()