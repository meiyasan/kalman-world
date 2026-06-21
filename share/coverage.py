#!/usr/bin/env python3
"""
Report functions that were never called according to the most recent coverage run.
Reads the .coverage data file written by pytest-cov / coverage.py.
"""
import ast
import os
import sys

# This file is named coverage.py, so its own directory on sys.path[0] would shadow the
# installed ``coverage`` package. Drop it before importing the real package.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]

try:
    import coverage as cov_module
except ImportError:
    print("ERROR: 'coverage' package not found – run: pip install pytest-cov", file=sys.stderr)
    sys.exit(1)

EXCLUDE_PATHS = ("tests", "vendor", ".egg", "site-packages")

# ANSI colour helpers (no-op when stdout is not a TTY)
_TTY = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text

def RED(t):    return _c("31", t)
def YELLOW(t): return _c("33", t)
def CYAN(t):   return _c("36", t)
def BOLD(t):   return _c("1",  t)
def DIM(t):    return _c("2",  t)
def GREEN(t):  return _c("32", t)


def _is_excluded(path: str) -> bool:
    return any(p in path for p in EXCLUDE_PATHS)


def _is_docstring(stmt) -> bool:
    """True if ``stmt`` is a bare string-literal expression (a docstring)."""
    return (isinstance(stmt, ast.Expr)
            and isinstance(getattr(stmt, "value", None), ast.Constant)
            and isinstance(stmt.value.value, str))


def _first_executable_line(node) -> int:
    """First *executable* body line, skipping a leading docstring.

    A docstring is compiled into ``__doc__`` at def-time and is never recorded as an
    executed line, so using ``body[0]`` blindly would flag every documented function as
    uncovered even when it ran.
    """
    body = node.body
    if body and _is_docstring(body[0]):
        body = body[1:]
    return body[0].lineno if body else node.body[0].lineno


def _uncovered_functions(fname: str, executed_lines: set) -> list[str]:
    try:
        with open(fname) as fh:
            source = fh.read()
        tree = ast.parse(source)
    except Exception:
        return []

    results = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # A function is "not covered" when its first executable line was never executed
            body_first_line = _first_executable_line(node)
            if body_first_line not in executed_lines:
                file_part = DIM(fname)
                func_part = YELLOW(node.name)
                line_part = DIM(f"line {node.lineno}")
                results.append(f"  {file_part}::{func_part}  ({line_part})")
    return results


def main() -> None:
    cov = cov_module.Coverage()
    try:
        cov.load()
    except cov_module.exceptions.NoDataError:
        print(RED("No coverage data found") + " – run 'make tests' first.")
        sys.exit(1)

    data = cov.get_data()
    uncovered: list[str] = []

    for fname in sorted(data.measured_files()):
        if _is_excluded(fname):
            continue
        executed = set(data.lines(fname) or [])
        uncovered.extend(_uncovered_functions(fname, executed))

    if uncovered:
        header = BOLD(RED(f"Uncovered functions ({len(uncovered)} total):"))
        print(f"\n{header}")
        for entry in uncovered:
            print(entry)
    else:
        print(f"\n{GREEN(BOLD('All measured functions are covered!'))}")


if __name__ == "__main__":
    main()
