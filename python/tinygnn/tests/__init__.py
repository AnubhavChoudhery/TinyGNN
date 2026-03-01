"""
tinygnn.tests — Smoke-test suite for post-installation verification.

Run via:
    python -m tinygnn.tests        # all tests
    python -m tinygnn.tests -v     # verbose (show each test name)
    tinygnn-test                   # CLI entry point (same as above)
"""
from .smoke_tests import run_all, TestResult

__all__ = ["run_all", "TestResult"]
