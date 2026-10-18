"""MiniQMT input adapter for the existing paired ETF snapshot updater."""
from pathlib import Path
import hashlib
import json

import pandas as pd

from data_provider.miniqmt_bridge import MiniQMTBridgeClient, MiniQMTError, canonical_bytes, verified_pair
from data_provider.miniqmt_fetcher import checked_bars, market_symbol
from quant_data_contract import sha256_file, DataContractError


def latest_session(through):
    finish = pd.Timestamp(through).normalize()
    start = finish - pd.Timedelta(days=31)
    response = MiniQMTBridgeClient().request("calendar", ["SH", "SZ"], start.date(), finish.date())
    days = response["data"]["SH"]["dates"]
    if not days or days != response["data"]["SZ"]["dates"]:
        raise MiniQMTError("MiniQMT trading calendars are empty or inconsistent")
    parsed = pd.to_datetime(days, errors="coerce")
    if parsed.isna().any() or parsed.duplicated().any() or any(not start <= d <= finish for d in parsed):
        raise MiniQMTError("invalid MiniQMT trading calendar")
    return parsed.max().normalize()


def verified_response(cache: Path, receipt: dict) -> tuple[Path, dict]:
    relative = receipt.get("relative_path", "")
    path = cache / relative
    if (not relative or Path(relative).is_absolute() or ".." in Path(relative).parts
            or any((cache / Path(*Path(relative).parts[:i])).is_symlink() for i in range(1, len(Path(relative).parts) + 1))
            or not path.is_file() or sha256_file(path) != receipt.get("sha256")):
        raise DataContractError("MiniQMT source response changed")
    value = json.loads(path.read_bytes())
    if (value.get("status") != "ok" or value.get("provider") != "miniqmt"
            or hashlib.sha256(canonical_bytes(value["data"])).hexdigest() != value.get("data_sha256")
            or hashlib.sha256(canonical_bytes(value["request"])).hexdigest() != value.get("request_sha256")):
        raise DataContractError("MiniQMT source response digest differs")
    return path, value


class MiniQMTEtfSource:
    """At most twenty same-window securities share one frozen bridge response."""
    def __init__(self, cache: Path, windows: dict):
        self.client = MiniQMTBridgeClient()
        self.cache = cache
        self.windows = dict(windows)
        self.batch = {}
        self.receipts = {}
        self.network_calls = None  # SDK-internal HTTP/RPC request count is not observable.
        self.bridge_calls = 0

    def close(self):
        self.batch.clear()

    def fetch_pair(self, symbol, metadata, *, start_date, end_date, request_gate=None, **_):
        if symbol not in self.batch:
            group = [s for s, window in self.windows.items() if window == (start_date, end_date)][:20]
            if symbol not in group:
                group = [symbol] + group[:19]
            codes = [market_symbol(s) for s in group]
            call = request_gate.call if request_gate is not None else lambda f, *a, **k: f(*a, **k)
            response = call(self.client.request, "update_pair", codes, start_date, end_date)
            self.bridge_calls += 1
            sha = hashlib.sha256(canonical_bytes(response)).hexdigest()
            relative = f"miniqmt-responses/{sha}.json"
            path = self.cache / relative
            if path.exists():
                if sha256_file(path) != sha:
                    raise DataContractError("MiniQMT source response changed")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                # Canonical bytes are the exact bridge payload accepted above.
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(canonical_bytes(response))
                temporary.replace(path)
            self.batch = {s: (response["data"][c], {"relative_path": relative, "sha256": sha})
                          for s, c in zip(group, codes)}
            for s in group:
                self.windows.pop(s, None)
        data, receipt = self.batch.pop(symbol)
        self.receipts[symbol] = receipt
        raw, qfq = verified_pair(data, start_date, end_date)
        raw, qfq = checked_bars(raw), checked_bars(qfq)
        if raw.empty or qfq.empty or raw["date"].tolist() != qfq["date"].tolist():
            raise MiniQMTError("MiniQMT raw/QFQ cache coverage is incomplete")
        # Validate cash/share facts against the SDK's independently adjusted
        # prices. Empty actions certify only this observed window, not history.
        for frame, component in [(raw, "raw"), (qfq, "qfq")]:
            frame["date"] = pd.to_datetime(frame["date"])
            frame["symbol"] = symbol
            frame["name"] = metadata["name"]
            frame["source"] = f"miniqmt.{component}"
            frame["source_id"] = f"miniqmt:response:{receipt['sha256']}"
            frame["provider_response_key"] = "explicit_cash_share_front" if component == "qfq" else "native_raw"
            frame.attrs["source_receipt"] = receipt
            frame.attrs["miniqmt_actions"] = data["actions"]
            frame.attrs["miniqmt_action_through"] = data.get("action_history_through", end_date)
        return raw, qfq
