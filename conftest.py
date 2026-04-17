"""
conftest.py — AeroNode test configuration.

This file registers the missing ROS launch_testing hook spec that is present
on this machine (ROS Jazzy is installed system-wide). Without this stub,
pytest's pluggy raises PluginValidationError during check_pending() because
launch_testing_ros registers 'pytest_launch_collect_makemodule' but the
hook specification is never declared (since we haven't imported all of ROS).
"""

import pytest


class _RosLaunchTestingHookSpecs:
    """Stub hook specs to satisfy launch_testing_ros_pytest_entrypoint."""

    @pytest.hookspec(firstresult=True)
    def pytest_launch_collect_makemodule(self, path, parent, entrypoint):  # noqa: D102
        """Stub for ROS launch_testing hook — never called in normal pytest runs."""


def pytest_configure(config: pytest.Config) -> None:
    """Register stub hook specs before check_pending() runs."""
    config.pluginmanager.add_hookspecs(_RosLaunchTestingHookSpecs)
