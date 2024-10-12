import time
import functools


# Decorator for logging information for functions.
def log_info(log_path: str = None, log_enabled: bool = False, print_enabled: bool = False,
             display_args: bool = True):
    """
    Control logging information for functions when called. Logging includes both printing to
    stdout, and saving logs to a file (at a provided path). Logs are appended to log
    files, so they accumulate over time.

    :param log_path:        path to log file (if applicable).
    :param log_enabled:     whether logging to files is enabled.
    :param print_enabled:   whether logging to stdout is enabled.
    :param display_args:    whether logging includes displaying explicit function call arguments.
    :return:                decorated function.
    """
    def log_info_decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            now = time.time()
            if display_args:
                string = (f"{now}: Called function {function.__qualname__} with following information:\n"
                          f"   args:   {[str(arg) for arg in args]}\n"
                          f"   kwargs: {[f'{str(kwarg)} = {str(kwargs[kwarg])}' for kwarg in kwargs]}")
            else:
                string = f"{now}: Called function {function.__qualname__}."
            # Log if log_enabled is True and log_path is provided:
            if (log_enabled is True) and (log_path is not None):
                with open(log_path, "a") as log_file:
                    log_file.write(string)
                    log_file.write("\n\n")
            # If log_enabled is True but no log_path is provided:
            elif (log_enabled is True) and (log_path is None):
                print(f"{now}: DecoratorWarn: @log_info used in function {function.__qualname__} but a "
                      f"logging path was not provided. Not logging information saved.")
            # If printing is turned on:
            if print_enabled:
                print(string)
            return function(*args, **kwargs)
        return wrapper
    return log_info_decorator
