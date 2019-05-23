from itertools import chain


def flatten(ls):
    """
    Flatten one level of nesting
    """
    return chain.from_iterable(ls)

