#!/usr/bin/python

import sys
import time
import psutil
import logging
import threading
import tracemalloc
import traceback
from pathlib import Path
from typing import Union
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# DECORATORS
def perf_monitor(func):
    """ Measure performance of a function """
    @wraps(func)
    def wrapper(*args, **kwargs):
        strt_time = time.perf_counter()
        cpu_percent_prev = psutil.cpu_percent(interval=0.05, percpu=False)
        tracemalloc.start()
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"Exception in {func.__name__}: {e}", exc_info=True, stack_info=True)
        finally:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=False)
            cpu_percnt = max(0, cpu_percent - cpu_percent_prev)  # Avoid negative values
            end_time = time.perf_counter()
            duration = end_time - strt_time
            msj = f"{func.__name__}\t\tUsed {cpu_percnt:>5.1f} % CPU: {hm_time(duration)}\t Mem: [avg:{hm_sz(current):>8}, max:{hm_sz(peak):>8}]\t({func.__doc__})"
            logging.info(msj)
    return wrapper

def show_running_message_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        message = f" {func.__name__} running"

        def progress_indicator():
            sys.stdout.write(message)
            sys.stdout.flush()
            while not progress_indicator.stop:
                for pattern in "|/-o+\\":
                    if progress_indicator.stop:
                        break
                    sys.stdout.write(f"\r{message} {pattern}")
                    sys.stdout.flush()
                    time.sleep(0.1)
            sys.stdout.write(f"\r{message} Done!\n")
            sys.stdout.flush()

        progress_indicator.stop = False
        progress_thread = threading.Thread(target=progress_indicator)
        progress_thread.start()

        try:
            result = func(*args, **kwargs)
        finally:
            progress_indicator.stop = True
            progress_thread.join()

        return result
    return wrapper

def measure_cpu_utilization(func):
    """Measure CPU utilization"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        cpu_count = psutil.cpu_count(logical=True)
        strt_time = time.monotonic()
        result = func(*args, **kwargs)
        cpu_prcnt = psutil.cpu_percent(interval=0.1, percpu=True)  # Measure after execution
        end_time = time.monotonic()
        avg_cpu_percent = sum(cpu_prcnt) / cpu_count
        return result, avg_cpu_percent, cpu_prcnt
    return wrapper

def perf_monitor_temp(func):
    """ Measure performance of a function with CPU temperature """
    @wraps(func)
    def wrapper(*args, **kwargs):
        strt_time = time.perf_counter()
        cpu_percent_prev = psutil.cpu_percent(interval=0.05, percpu=False)
        tracemalloc.start()
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"Exception in {func.__name__}: {e}", exc_info=True, stack_info=True)
        finally:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=False)
            cpu_percnt = max(0, cpu_percent - cpu_percent_prev)  # Avoid negative values

            # Safe check for temperature sensor availability
            cpu_temp = None
            try:
                sensors = psutil.sensors_temperatures()
                if "coretemp" in sensors and sensors["coretemp"]:
                    cpu_temp = sensors["coretemp"][0].current
                    logging.info(f"CPU temperature: {cpu_temp}°C")
            except AttributeError:
                logging.warning("CPU temperature monitoring not supported on this system.")

            end_time = time.perf_counter()
            duration = end_time - strt_time
            msj = f"{func.__name__}\t\tUsed {cpu_percnt:>5.1f} % CPU: {hm_time(duration)}\t Mem: [avg:{hm_sz(current):>8}, max:{hm_sz(peak):>8}]\t({func.__doc__})"
            logging.info(msj)
    return wrapper

def hm_sz(numb: Union[str, int, float], type: str = "B") -> str:
    '''Convert file size to human-readable format'''
    numb = float(numb)
    if numb < 1024.0:
        return f"{numb:.2f} {type}"
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E']:
        if numb < 1024.0:
            return f"{numb:.2f} {unit}{type}"
        numb /= 1024.0
    return f"{numb:.2f} {unit}{type}"

def hm_time(timez: float) -> str:
    '''Convert time to human-readable format'''
    units = {
        'year': 31536000,
        'month': 2592000,
        'week': 604800,
        'day': 86400,
        'hour': 3600,
        'min': 60,
        'sec': 1,
    }
    if timez < 0:
        return "Error: negative time"
    elif timez == 0:
        return "Zero"
    elif timez < 0.001:
        return f"{timez * 1000:.3f} ms"
    elif timez < 60:
        return f"{timez:.3f} sec"
    
    frmt = []
    for unit, seconds_per_unit in units.items():
        value = timez // seconds_per_unit
        if value != 0:
            frmt.append(f"{int(value)} {unit}{'s' if value > 1 else ''}")
        timez %= seconds_per_unit
    
    return ", ".join(frmt[:-1]) + " and " + frmt[-1] if len(frmt) > 1 else frmt[0] if frmt else "0 sec"

def file_size(path):
    """Return file/dir size in MB"""
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    return 0.0

# Example Usage
@perf_monitor
def example_function():
    """Example function for testing performance monitoring"""
    time.sleep(1)  # Simulate work

if __name__ == "__main__":
    example_function()