"""Thin setup.py shim: pyproject.toml carries all declarative metadata; this file
exists only to wire up cffi's setuptools integration (``cffi_modules``), which has
no declarative pyproject.toml equivalent. Building the C extension is optional --
see ``modules/framework/native/__init__.py`` for the pure-Python fallback.
"""
from setuptools import setup

setup(
    cffi_modules=["modules/framework/native/cffi.py:ffi"],
)
