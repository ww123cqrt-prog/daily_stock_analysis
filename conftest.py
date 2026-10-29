"""Private runtime paths and offline sockets shared by both Python suites."""

import ipaddress
import os
import socket
import tempfile
from pathlib import Path

import dotenv
import pytest


_REPO_ROOT = Path(__file__).resolve().parent
_SESSION_RUNTIME = tempfile.TemporaryDirectory(prefix="dsa-pytest-")
_EXTERNAL_PATH_ENV = (
    "SCREENING_ENV_FILE", "SCREENING_ENV_FILES", "QLIB_PROVIDER_URI",
    "SCREENING_DAILY_HISTORY_CACHE_DIR", "DAILY_HISTORY_CACHE_DIR",
    "SCREENING_INDUSTRY_PROVIDER_CACHE_DIR", "INDUSTRY_PROVIDER_CACHE_DIR",
    "MLFLOW_TRACKING_URI", "LITELLM_CONFIG",
)


def _runtime_environment(root: Path) -> dict[str, str]:
    env_file = root / ".env"
    env_file.write_text("", encoding="utf-8")
    return {
        "ENV_FILE": str(env_file),
        "DATABASE_PATH": str(root / "stock_analysis.db"),
        "LOG_DIR": str(root / "logs"),
        "DSA_STRATEGY_ASSETS_DB": str(root / "strategy_assets.db"),
        "QUANT_WORKFLOW_REPORTS_ROOT": str(root / "quant_workflow" / "reports"),
        "QUANT_WORKFLOW_MLRUNS_DIR": str(root / "quant_workflow" / "mlruns"),
        "QUANT_DATA_ROOT": str(root / "quant_data"),
        "SCREENING_DATA_DIR": str(root / "screening"),
        "MINIQMT_CONFIG_DIR": str(root / "miniqmt-config"),
        "MINIQMT_TRANSPORT": "ssh",
    }


# Set these before test-module imports can construct service singletons.
os.environ.update(_runtime_environment(Path(_SESSION_RUNTIME.name)))
for _key in _EXTERNAL_PATH_ENV:
    os.environ.pop(_key, None)
for _key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_key, None)


_original_load_dotenv = dotenv.load_dotenv
_original_dotenv_values = dotenv.dotenv_values


def _private_dotenv_path(path):
    if path is None or Path(path).resolve() == _REPO_ROOT / ".env":
        configured = os.environ.get("ENV_FILE")
        if configured and Path(configured).resolve() != _REPO_ROOT / ".env":
            return configured
        return str(Path(_SESSION_RUNTIME.name) / ".env")
    return path


def _load_dotenv(dotenv_path=None, *args, **kwargs):
    path = dotenv_path if kwargs.get("stream") is not None else _private_dotenv_path(dotenv_path)
    return _original_load_dotenv(path, *args, **kwargs)


def _dotenv_values(dotenv_path=None, *args, **kwargs):
    path = dotenv_path if kwargs.get("stream") is not None else _private_dotenv_path(dotenv_path)
    return _original_dotenv_values(path, *args, **kwargs)


dotenv.load_dotenv = _load_dotenv
dotenv.dotenv_values = _dotenv_values


_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_allow_network = False


def _check_offline_address(address):
    if _allow_network:
        return
    if not isinstance(address, tuple):  # Unix-domain sockets stay local.
        return
    host = address[0]
    if isinstance(host, bytes):
        host = host.decode("ascii")
    if str(host).lower() == "localhost":
        return
    try:
        if ipaddress.ip_address(str(host).split("%", 1)[0]).is_loopback:
            return
    except ValueError:
        pass
    raise AssertionError(f"offline test attempted a non-loopback connection: {host}")


def _offline_connect(self, address):
    _check_offline_address(address)
    return _original_connect(self, address)


def _offline_connect_ex(self, address):
    _check_offline_address(address)
    return _original_connect_ex(self, address)


# Import-time clients are subject to the same boundary as test bodies.
socket.socket.connect = _offline_connect
socket.socket.connect_ex = _offline_connect_ex


def _refuse_live_miniqmt(*args, **kwargs):
    raise AssertionError("offline test attempted a live MiniQMT SSH/SDK request")


@pytest.fixture
def make_symlink():
    """Create a real link, or report the specific missing Windows privilege."""
    def create(link: Path, target: Path, *, target_is_directory: bool = False):
        try:
            link.symlink_to(target, target_is_directory=target_is_directory)
        except OSError as exc:
            if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
                pytest.skip("Windows symlink privilege unavailable (WinError 1314)")
            raise
    return create


@pytest.fixture(autouse=True)
def _isolate_test_runtime(tmp_path, monkeypatch, request):
    global _allow_network
    _allow_network = (
        request.config.getoption("markexpr").strip() == "network"
        and request.node.get_closest_marker("network") is not None
    )
    monkeypatch.chdir(tmp_path)
    for key, value in _runtime_environment(tmp_path).items():
        monkeypatch.setenv(key, value)
    for key in (*_EXTERNAL_PATH_ENV, "QMT_PYTHON", "QMT_PORT", "QMT_DATA_DIR", "QMT_TIMEOUT", "QMT_DATASET_DIR",
                "QMT_TRANSPORT", "QMT_REMOTE_TARGET", "QMT_DOWNLOAD_MISSING", "QMT_MIRROR_DIR",
                "QMT_ADJUSTMENT", "QMT_STOCK_ADJUSTMENT", "QMT_ETF_ADJUSTMENT"):
        monkeypatch.delenv(key, raising=False)
    from src.config import Config
    from src import auth
    from src.storage import DatabaseManager
    from src.repositories import strategy_assets_repo
    from src.services.screening import config as screening_config
    from data_provider import miniqmt_bridge

    monkeypatch.setattr(miniqmt_bridge, "_ssh_request", _refuse_live_miniqmt)
    try:
        from quant_qmt_data import QMTClient
    except ModuleNotFoundError as exc:
        if exc.name != "quant_qmt_data":
            raise
    else:
        original_request = QMTClient._request
        def offline_request(client, *args, **kwargs):
            if client.transport == 'mirror':
                return original_request(client, *args, **kwargs)
            return _refuse_live_miniqmt(*args, **kwargs)
        monkeypatch.setattr(QMTClient, "_request", offline_request)

    # Screening also searches its source root independently of ENV_FILE.
    monkeypatch.setattr(screening_config, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(strategy_assets_repo, "_default_repo", None)
    Config.reset_instance()
    DatabaseManager.reset_instance()
    auth.refresh_auth_state()
    yield
    _allow_network = False
    DatabaseManager.reset_instance()
    Config.reset_instance()
    strategy_assets_repo._default_repo = None
