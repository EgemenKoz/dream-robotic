import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gazebo: requires a live Gazebo simulation (skipped in unit-test runs)",
    )
