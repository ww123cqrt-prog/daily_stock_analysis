"""Windows liveness checks must never generate console-control signals."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import psutil
import pytest


PROBES = (
    ("quant_workflow.scripts._daily_update_runtime", "pid_is_alive"),
    ("quant_workflow.scripts.launch_robust_strategy", "_pid_alive"),
    ("src.agent.codex_tool_process", "_pid_alive"),
)


@pytest.mark.parametrize("module_name,function_name", PROBES)
@pytest.mark.parametrize("state,expected", (
    ("alive", True), ("dead", False), ("missing", False),
    ("denied", True), ("unreadable", True), ("no_psutil", True),
))
def test_windows_pid_queries_are_read_only_and_fail_closed(module_name, function_name, state, expected):
    module = importlib.import_module(module_name)
    process = SimpleNamespace(is_running=Mock(return_value=state != "dead"))
    factory = Mock(return_value=process)
    if state == "missing":
        factory.side_effect = psutil.NoSuchProcess(424242)
    elif state == "denied":
        factory.side_effect = psutil.AccessDenied(424242)
    elif state == "unreadable":
        process.is_running.side_effect = OSError("query unavailable")
    backend = SimpleNamespace(Process=factory, NoSuchProcess=psutil.NoSuchProcess, Error=psutil.Error)
    kill = Mock(side_effect=AssertionError("Windows PID query must not signal"))
    with patch.object(module, "os", SimpleNamespace(name="nt", kill=kill)), patch.dict(
        "sys.modules", {"psutil": None if state == "no_psutil" else backend}
    ):
        observed = getattr(module, function_name)(424242)
    assert observed is expected
    kill.assert_not_called()


@pytest.mark.parametrize("module_name,function_name", PROBES)
@pytest.mark.parametrize("pid", (0, -1))
def test_windows_pid_queries_reject_nonpositive_pid_without_signalling(module_name, function_name, pid):
    module = importlib.import_module(module_name)
    kill = Mock(side_effect=AssertionError("nonpositive PID must not signal"))
    with patch.object(module, "os", SimpleNamespace(name="nt", kill=kill)):
        observed = getattr(module, function_name)(pid)
    assert observed is False
    kill.assert_not_called()
