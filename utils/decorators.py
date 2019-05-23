import time
from functools import wraps


def timeit(f):
    @wraps(f)
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f"'{f.__name__}' {round(te - ts, 2)} s")
        return result
    return timed

