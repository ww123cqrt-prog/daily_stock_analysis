"""Checks for the offline suite's shared runtime isolation."""

import os
import socket
from pathlib import Path

import pytest


def test_runtime_paths_are_private(tmp_path):
    root = tmp_path.resolve()
    assert Path.cwd().resolve() == root
    for key in ("ENV_FILE", "DATABASE_PATH", "QUANT_WORKFLOW_REPORTS_ROOT",
                "QUANT_WORKFLOW_MLRUNS_DIR", "QUANT_DATA_ROOT"):
        assert Path(os.environ[key]).resolve().is_relative_to(root), key
    assert Path(os.environ["ENV_FILE"]).is_file()


def test_offline_guard_rejects_external_connections():
    with socket.socket() as client:
        with pytest.raises(AssertionError, match="offline test"):
            client.connect(("192.0.2.1", 443))


def test_offline_guard_keeps_real_loopback_connections():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=1):
            connection, _ = server.accept()
            connection.close()


def test_default_database_and_assets_use_private_paths():
    from src.repositories.strategy_assets_repo import StrategyAssetsRepository
    from src.storage import DatabaseManager

    assert DatabaseManager()._db_url == f"sqlite:///{os.environ['DATABASE_PATH']}"
    assert StrategyAssetsRepository()._db_path == Path(os.environ["DSA_STRATEGY_ASSETS_DB"])


def test_default_miniqmt_transport_cannot_start_ssh():
    from data_provider.miniqmt_bridge import MiniQMTBridgeClient

    with pytest.raises(AssertionError, match="live MiniQMT"):
        MiniQMTBridgeClient().request("calendar", ["SH"], "2026-10-16", "2026-10-16")


def test_package_miniqmt_constructor_reads_only_the_private_env(tmp_path, monkeypatch):
    import dotenv
    from quant_qmt_data.miniqmt_bridge import MiniQMTBridgeClient

    original = dotenv.load_dotenv
    observed = []

    def load_private(path, **kwargs):
        assert Path(path).resolve() == (tmp_path / ".env").resolve()
        observed.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(dotenv, "load_dotenv", load_private)
    MiniQMTBridgeClient()
    assert len(observed) == 1
