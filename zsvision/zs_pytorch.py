# -* coding: utf-8 -*
import torch
import time

def format_time(time_secs):
    return '{0:.3f}s'.format(time_secs) # can upgrade later

def start_timer():
    """useful for benchmarking"""
    torch.cuda.synchronize()
    return time.perf_counter()

def summary(name, start_time, display=False):
    """useful for benchmarking"""
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    step_time = end_time - start_time
    if display:
        print('{} time: {}'.format(name, format_time(step_time)))
    return step_time
