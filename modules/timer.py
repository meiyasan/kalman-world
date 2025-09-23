import atexit
import os, sys
import time
import importlib
from collections import defaultdict
from .logging import Logger

logger = Logger(name="Import Timer")

class ImportTimer:
    _instance = None
    _import_times = {}

    _eager_loading = False
    _default_depth = 0
    _threshold = 0
    _filter_names = ["*"]
    _limit = None

    def __new__(cls, *args, **kwargs):
        """Ensure singleton instance"""
        if cls._instance is None:
            logger.debug("Starting import timer. Report will be provided at the end of the program.")
            cls._instance = super().__new__(cls)
            cls._instance.execution_time = time.perf_counter()
            sys.meta_path.insert(0, cls._instance)  
            atexit.register(cls._instance.cleanup)  

        # Update parameters
        cls._filter_names = list(args) if args else cls._filter_names
        cls._default_depth = kwargs.get("depth", cls._default_depth)
        cls._threshold = kwargs.get("threshold", cls._threshold)
        cls._eager_loading = kwargs.get("eager", cls._eager_loading)
        cls._limit = kwargs.get("limit", cls._limit)

    def _start(self):
        """Start timing an import."""
        self.start_time = time.perf_counter()

    def _stop(self, package_name):
        """Stop timing and store import time."""
        if self.start_time:
            elapsed = (time.perf_counter() - self.start_time) * 1e6  # Convert to µs
            ImportTimer._import_times[package_name] = ImportTimer._import_times.get(package_name, 0) + elapsed
            self.start_time = None

    def find_spec(self, name, path, target=None):
        """Hook into module imports to measure time."""
        self._start()
        spec = importlib.machinery.PathFinder.find_spec(name, path, target)

        if spec and ImportTimer._eager_loading:
            try:
                sys.meta_path.remove(self)  
                importlib.import_module(name)
            finally:
                sys.meta_path.insert(0, self)  

        self._stop(name)
        return spec

    @staticmethod
    def print(depth=-1):
        """Display import times with optional depth grouping."""
        depth = depth if depth >= 0 else ImportTimer._default_depth
        grouped_times = defaultdict(float)

        for package, time_elapsed in ImportTimer._import_times.items():
            key = ".".join(package.split(".")[:depth+1]) if depth is not None else package
            grouped_times[key] += time_elapsed  

        sorted_imports = sorted(grouped_times.items(), key=lambda x: -x[1])
        packages = {pkg.split('.')[0] for pkg, _ in sorted_imports}
        total_time = sum(time_elapsed for _, time_elapsed in sorted_imports)
        total_execution = (time.perf_counter() - ImportTimer._instance.execution_time) * 1e6  

        logger.debug(f"Execution Time: {total_execution/1e3:.2f} ms")
        logger.debug(f"Total Import Time: {total_time/1e3:.2f} ms across {len(ImportTimer._import_times)} imports from {len(packages)} packages")

        if ImportTimer._filter_names != ["*"]:
            logger.debug(f"Filtered to: {', '.join(ImportTimer._filter_names)}")
        if ImportTimer._threshold:
            unit = "ms" if ImportTimer._eager_loading else "µs"
            threshold = ImportTimer._threshold / 1e3 if ImportTimer._eager_loading else ImportTimer._threshold
            logger.debug(f"Threshold: {threshold} {unit}")

        N = 0
        for package, elapsed in sorted_imports:
            if ImportTimer._threshold and elapsed < ImportTimer._threshold:
                continue 
            if ImportTimer._filter_names != ["*"] and not any(package.startswith(f"{f}.") or package == f for f in ImportTimer._filter_names):
                continue
            if ImportTimer._limit and N >= ImportTimer._limit:
                break

            elapsed_display = elapsed / 1e3 if ImportTimer._eager_loading else elapsed
            logger.warning(f"Package '{package}' imported in {elapsed_display:.2f} {'ms' if ImportTimer._eager_loading else 'µs'}")
            N += 1

    def cleanup(self):
        """Print report at exit."""
        self.print()
        sys.meta_path.remove(self)  

# Initialize singleton
def start(**kwargs):
    return ImportTimer(**kwargs)

def stop():
    return ImportTimer().cleanup()

class Timer(ImportTimer):
    pass

# Example usage:
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Show import times at different depths
    ImportTimer.display_import_times(depth=0)  # Group by top-level packages
    ImportTimer.display_import_times(depth=1)  # Group by package + first submodule
    ImportTimer.display_import_times(depth=None)  # Show all details