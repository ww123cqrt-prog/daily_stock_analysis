"""Flush-only JSONL checkpoints for explicitly enabled pytest sessions."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--checkpoint-log",
        metavar="PATH",
        default=None,
        help="Exclusively create a JSONL event log for one pytest session.",
    )
    parser.addoption('--checkpoint-failures', default=None,
                     help='Select and verify the explicit first-round failure manifest before execution.')
    parser.addoption('--checkpoint-sources', default=None,
                     help='Frozen source identity used to reject unlisted tests before their bodies run.')


def pytest_collection_modifyitems(config, items):
    source = config.getoption('checkpoint_sources')
    if source is None:
        return
    try:
        frozen = json.loads(Path(source).read_text(encoding='utf-8'))
        root = Path(frozen['source_root']).resolve(strict=True)
        if root != Path(config.rootpath).resolve():
            raise ValueError('source root differs from this checkout')
        allowed = set(frozen['files'])
        for item in items:
            path = Path(item.path).resolve(strict=True)
            if not path.is_relative_to(root) or path.relative_to(root).as_posix() not in allowed:
                raise ValueError(f'collected test absent from source manifest: {item.nodeid}')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise pytest.UsageError(f'invalid checkpoint source manifest: {exc}') from exc


def pytest_configure(config):
    path = config.getoption("checkpoint_log")
    failures = config.getoption('checkpoint_failures')
    if failures and (path is None or getattr(config.option, 'numprocesses', None)):
        raise pytest.UsageError('explicit failure confirmation requires a serial checkpoint session')
    if path is None:
        return
    if hasattr(config, "workerinput"):
        return
    try:
        stream = Path(path).open("x", encoding="utf-8", newline="\n")
    except FileExistsError as error:
        raise pytest.UsageError(f"checkpoint log already exists: {path}") from error
    plugin = _CheckpointLog(stream, {
        key: os.environ.get("QUANT_TEST_" + key.upper())
        for key in ("run_id", "block_id", "segment_id")
    }, failures)
    try:
        config.pluginmanager.register(plugin, "checkpoint-log")
    except BaseException:
        stream.close()
        raise


def _utc():
    return datetime.now(timezone.utc).isoformat()


class _CheckpointLog:
    def __init__(self, stream, identity=None, failures=None):
        self.stream = stream
        self.identity = identity or {}
        self.expected = None
        if failures:
            from scripts.testing.block_runner import normalize_nodeids
            try:
                manifest = json.loads(Path(failures).read_text(encoding='utf-8'))
                self.source_root = Path(manifest['source_root']).resolve(strict=True)
                self.expected = manifest['nodeids']
                if (not isinstance(self.expected, list) or not self.expected
                        or normalize_nodeids(self.source_root, manifest['first_failed']) != self.expected
                        or normalize_nodeids(self.source_root, self.expected) != self.expected):
                    raise ValueError('failure manifest is empty or inconsistent')
                if Path.cwd().resolve() != self.source_root:
                    raise ValueError('failure manifest source root differs from execution cwd')
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                stream.close()
                raise pytest.UsageError(f'invalid failure manifest: {exc}') from exc

    def _selection(self, items):
        from scripts.testing.block_runner import normalize_nodeids
        try:
            return normalize_nodeids(self.source_root, [item.nodeid for item in items])
        except (OSError, ValueError) as exc:
            self._reject_selection(str(exc))

    def _reject_selection(self, reason):
        self._emit('selection_error', error=reason)
        raise pytest.UsageError(f'rerun selection differs from first failures: {reason}')

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, session, config, items):
        yield
        if self.expected is not None:
            nodes = self._selection(items)
            wanted = set(self.expected)
            selected = [item for item, node in zip(items, nodes) if node in wanted]
            deselected = [item for item, node in zip(items, nodes) if node not in wanted]
            if {node for node in nodes if node in wanted} != wanted:
                self._reject_selection('missing first-round failures')
            items[:] = selected
            config.hook.pytest_deselected(items=deselected)

    def _emit(self, event, **fields):
        self.stream.write(json.dumps(
            {"event": event, **self.identity, **fields, "utc": _utc(), "monotonic": time.monotonic()},
            ensure_ascii=False, allow_nan=False,
        ) + "\n")
        self.stream.flush()

    def _close(self):
        if not self.stream.closed:
            self.stream.close()

    def pytest_sessionstart(self, session):
        self._emit(
            "session_start", schema_version=1, pid=os.getpid(),
            python_version=platform.python_version(), pytest_version=pytest.__version__,
            cwd=str(Path.cwd()), rootpath=str(session.config.rootpath),
        )

    @pytest.hookimpl(trylast=True)
    def pytest_collection_finish(self, session):
        if not getattr(session.config.option, "numprocesses", None):
            self._emit("collection", nodeids=[item.nodeid for item in session.items])
            if self.expected is not None and set(self._selection(session.items)) != set(self.expected):
                self._reject_selection('selected collection does not equal first-round failures')

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtestloop(self, session):
        if self.expected is not None and set(self._selection(session.items)) != set(self.expected):
            self._reject_selection('execution collection does not equal first-round failures')
        return (yield)

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids):
        self._emit("collection", nodeids=ids)

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node, error):
        if error:
            self._emit("worker_error", error=str(error))

    def pytest_internalerror(self, excrepr, excinfo):
        self._emit("internal_error", error=str(excrepr))

    def pytest_collectreport(self, report):
        if report.failed:
            self._emit(
                "collection_error", nodeid=report.nodeid,
                longrepr=str(report.longrepr),
            )

    def pytest_runtest_logstart(self, nodeid, location):
        self._emit("node_start", nodeid=nodeid)

    def pytest_runtest_logreport(self, report):
        fields = {
            "nodeid": report.nodeid,
            "when": report.when,
            "outcome": report.outcome,
            "duration": report.duration,
            "wasxfail": getattr(report, "wasxfail", None),
            "xpass_strict": report.failed and str(report.longrepr).startswith("[XPASS(strict)]"),
        }
        if report.failed or report.skipped:
            fields["longrepr"] = str(report.longrepr)
        # pytest 9 sends genuine child reports through this same hook. Avoid
        # importing its new module so ordinary pytest 8 reports remain supported.
        if type(report).__name__ in {"SubtestReport", "SubTestReport"}:
            fields["context"] = str(getattr(report, "context", ""))
            fields["longrepr"] = str(report.longrepr)
            self._emit("subtest_report", **fields)
        else:
            self._emit("phase_report", **fields)

    def pytest_runtest_logfinish(self, nodeid, location):
        self._emit("node_finish", nodeid=nodeid)

    def pytest_sessionfinish(self, session, exitstatus):
        try:
            self._emit("session_finish", exitstatus=int(exitstatus))
        finally:
            self._close()

    def pytest_unconfigure(self, config):
        self._close()
