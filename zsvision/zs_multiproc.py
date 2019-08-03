"""Utilities for multiprocessing with Python.

NOTE: The starmap kwargs mapper is based on isarandi's StackOverflow answer here:
stackoverflow.com/questions/45718523/pass-kwargs-to-starmap-while-using-pool-in-python
"""
from itertools import repeat


def apply_kwargs(func, kwargs):
    """Wrapper for unpacking keyword function calls.

    Args:
        func (function): the function to be applied.
        kwargs (dict): the keywords and arguments for the function call.
    """
    return func(**kwargs)


def starmap_with_kwargs(pool, func, kwargs_iter):
    """Apply starmap with keyword arguments.

    Args:
        pool (multiprocessing.Pool): A pool of processes.
        func (function): the function to be applied.
        kwargs_iter (list[dict]): a list of dictionaries, each of which contains the
            keywords for a separate function call.
    """
    args_for_starmap = zip(repeat(func), kwargs_iter)
    return pool.starmap(apply_kwargs, args_for_starmap)
