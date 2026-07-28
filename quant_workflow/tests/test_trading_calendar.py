import importlib.util
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_module():
    path = SCRIPTS / "_trading_calendar.py"
    spec = importlib.util.spec_from_file_location("_trading_calendar_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_calendar_uses_canonical_native_provider(tmp_path, monkeypatch):
    module = load_module()
    provider = tmp_path / "native-provider"
    calendar = provider / "calendars" / "day.txt"
    calendar.parent.mkdir(parents=True)
    calendar.write_text("2026-07-23\n2026-07-24\n", encoding="utf-8")
    calls: list[str] = []

    def fake_provider_uri(provider_name: str = "canonical") -> Path:
        calls.append(provider_name)
        return provider

    monkeypatch.setattr(module, "qlib_provider_uri", fake_provider_uri)

    result = module.load_trading_calendar()

    assert calls == ["canonical"]
    assert result.path == calendar
    assert [item.isoformat() for item in result.days] == [
        "2026-07-23",
        "2026-07-24",
    ]
