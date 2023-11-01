from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        if func.__name__ not in TimeCounter.execution_time:
            TimeCounter.execution_time[func.__name__] = 0
        TimeCounter.execution_time[func.__name__] += total_time
        return result
    return timeit_wrapper

class TimeCounter(object):
    execution_time = {}