"""Build shim. The package is described by pyproject.toml; this file exists so
that ``pip install -e .`` works with every setuptools that still looks for it.

The Python package builds no extension of its own: its native kernels are the
C++ library (``make build install``), reached through cffi in ABI mode at run
time and optional -- see ``modules/framework/native/``.
"""
from setuptools import setup

setup()
