import concurrent.futures
import functools
import importlib
import pkgutil
import time
from itertools import product
from typing import Any

from jinja2 import BaseLoader, Environment
from requests import Response

from ecologits.tracers.utils import compute_llm_impacts, electricity_mixes


#
# String utils
#


def render_jinja(template: str, **kwargs):
    env = Environment(loader=BaseLoader())
    t = env.from_string(template)
    return t.render(**kwargs)


#
# Time utils
#


class Timer:
    """Usage

    with Timer() as timer:
        some_function()

    print(f"The function took {timer.execution_time} seconds to execute.")
    """

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time


#
# API utils
#


def retry(tries: int = 3, delay: int = 2):
    """
    A simple retry decorator that catch exception to retry multiple times
    @TODO: only catch only network error/timeout error.

    Parameters:
    - tries: Number of total attempts.
    - delay: Delay between retries in seconds.
    """

    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = tries
            while attempts > 1:
                try:
                    return func(*args, **kwargs)
                # @TODO: Catch network error.
                # except (requests.exceptions.RequestException, httpx.RequestError) as e:
                except Exception as e:
                    print(f"Error: {e}, retrying in {delay} seconds...")
                    time.sleep(delay)
                    attempts -= 1
            # Final attempt without catching exceptions
            return func(*args, **kwargs)

        return wrapper

    return decorator_retry


def log_and_raise_for_status(response: Response, msg_on_error: str = "API Error detail"):
    # response from requests module
    if not response.ok:
        try:
            error_detail = response.json().get("detail")
        except Exception:
            error_detail = response.text
        print(f"{msg_on_error}: {error_detail}\n")
        response.raise_for_status()


#
# Modules
#


def import_classes(package_name: str, class_names: list[str], more: list[str] = None) -> list[dict]:
    """Get a list of class obj from given package name and class_names.
    If `more` is given, it tries to extract the object with that names in the same module where a class is found.
    """
    # Import the package
    package = importlib.import_module(package_name)

    # Iterate over all modules in the package
    classes = []
    remaining_classes = set(class_names)
    for finder, name, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        # Import the module
        try:
            module = importlib.import_module(name)
        except Exception as e:
            print(f"Failed to import module {name}: {e}")
            continue

        # Check for each class in the module
        found_classes = remaining_classes.intersection(dir(module))
        for class_name in found_classes:
            cls = getattr(module, class_name)
            class_info = {"name": class_name, "obj": cls}
            for extra in more or []:
                class_info[extra] = getattr(module, extra, None)
            classes.append(class_info)
            remaining_classes.remove(class_name)

        # Stop if all classes have been found
        if not remaining_classes:
            break

    if remaining_classes:
        raise ValueError(f"Warning: The following classes were not found: {remaining_classes}")

    # Reorder the list of class
    class_indexes = {name: index for index, name in enumerate(class_names)}
    classes = sorted(classes, key=lambda d: class_indexes[d["name"]])

    return classes


#
# Async utils
#


def run_with_timeout(func, timeout, *args, **kwargs):
    """Set a timeout in seconds before stopping execution."""
    # @DEBUG: generates OSError: [Errno 24] Too many open files
    #         + uncatchable exception
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            print("Function execution exceeded the timeout.")
            return None


#
# Misc
#


def build_param_grid(common_params: dict[str, Any], grid_params: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """
    # Example usage:
    common_params = {
        "batch_size": 32,
        "model_params": {
            "dropout": 0.5,
            "activation": "relu"
        }
    }

    grid_params = {
        "learning_rate": [0.001, 0.01],
        "model_params": [
            {"hidden_layers": 2},
            {"hidden_layers": 3}
        ]
    }

    result = build_param_grid(common_params, grid_params)

    # Example of one entry in the result:
    # {
    #     "batch_size": 32,
    #     "learning_rate": 0.001,
    #     "model_params": {
    #         "dropout": 0.5,
    #         "activation": "relu",
    #         "hidden_layers": 2
    #     }
    # }
    """
    # Get all possible combinations of grid parameters
    keys = grid_params.keys()
    values = grid_params.values()
    combinations = list(product(*values))

    param_grid = []

    for combo in combinations:
        # Create a new dictionary starting with common_params
        params = common_params.copy()

        # Create dictionary for current combination
        current_combo = dict(zip(keys, combo))

        # For each parameter in the current combination
        for key, value in current_combo.items():
            if key in params and isinstance(params[key], dict) and isinstance(value, dict):
                # If both are dicts, merge at first level only
                params[key] = {**params[key], **value}
            else:
                # Otherwise, simply update the value
                params[key] = value

        param_grid.append(params)

    return param_grid


#
# carbon emission in kgCO2e (use Ecologits for estimation)
#


def impact_carbon(zone, token_count, request_latency):
    electricity_mix_zone = zone  # "FRA"#"WOR"
    electricity_mix = electricity_mixes.find_electricity_mix(zone=electricity_mix_zone)
    if_electricity_mix_adpe = electricity_mix.adpe
    if_electricity_mix_pe = electricity_mix.pe
    if_electricity_mix_gwp = electricity_mix.gwp

    model_active_parameter_count = 100
    model_total_parameter_count = 100

    return compute_llm_impacts(
        model_active_parameter_count=model_active_parameter_count,  # Nombre de paramètres activés pendant l'inférence. Pour les modèles denses, cette valeur est égale à model_total_parameter_count
        model_total_parameter_count=model_total_parameter_count,  # Nombre total de paramètres du modèle
        output_token_count=token_count,  # Nombre de tokens générés, disponible dans la réponse de l'API LLM
        if_electricity_mix_adpe=if_electricity_mix_adpe,
        if_electricity_mix_pe=if_electricity_mix_pe,
        if_electricity_mix_gwp=if_electricity_mix_gwp,
        request_latency=request_latency,  # temps de réponse
    )
