"""Bounded market-only access to the existing MiniQMT bridge over SSH.

Connection information stays in the workstation's private configuration.  No
SDK, trading module, HTTP service or account credentials are needed on the Mac.
"""
from __future__ import annotations

import base64
from datetime import date, datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Callable

import pandas as pd
import numpy as np


class MiniQMTError(RuntimeError):
    pass


def canonical_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def _day(value) -> date:
    return datetime.strptime(str(value).replace("-", ""), "%Y%m%d").date()


def _private_directory() -> Path:
    return Path(os.getenv("MINIQMT_CONFIG_DIR", str(Path.home() / ".agents/private/remote-windows-workstation"))).expanduser()


def _ps(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _ssh_request(request: dict, timeout: float) -> dict:
    """Invoke only the checked-in market bridge, with an owned remote deadline."""
    import yaml

    if not MiniQMTBridgeClient.configured():
        raise MiniQMTError("MiniQMT is disabled or its workstation profile is unavailable")
    config = _private_directory()
    target = os.getenv("MINIQMT_TARGET", "HOME")
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", target):
        raise MiniQMTError("invalid MiniQMT workstation profile")
    try:
        inventory = yaml.safe_load((config / "machine_configuration.yaml").read_text())["machines"][target]
        environment = subprocess.check_output(
            ["zsh", "-c", 'set -a; source "$1"; env -0', "zsh", str(config / "connection-targets.env")],
            timeout=3,
        ).decode()
        env = dict(v.split("=", 1) for v in environment.split("\0") if "=" in v)
        prefix = target + "_REMOTE_WIN_"
        py = inventory["software"]["python"]["exe"]
        pwsh = inventory["software"]["powershell7"]["exe"]
        root = inventory["projects"]["qmt_strategy_bridge"]["root"]
        port = int(inventory["software"]["xtquant"]["market_port"])
        if not 1 <= port <= 65535 or env.get(prefix + "AUTH_MODE", "key") != "key":
            raise ValueError("a verified market port and SSH key profile are required")
        host = env[prefix + "PREFERRED_HOST"]
        destination = env[prefix + "SSH_USER"] + "@" + host
        args = ["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=8",
                "-p", env[prefix + "SSH_PORT"]]
        if env.get(prefix + "SSH_IDENTITY_FILE"):
            args += ["-i", str(Path(env[prefix + "SSH_IDENTITY_FILE"]).expanduser())]
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        raise MiniQMTError("MiniQMT private workstation configuration is unavailable") from None

    # stdin is closed before waiting.  The bridge itself owns its SDK worker's
    # Job Object; this outer bound also cleans up a failed bridge invocation.
    script = """$ErrorActionPreference='Stop';[Console]::OutputEncoding=[Text.Encoding]::UTF8;
$si=[Diagnostics.ProcessStartInfo]::new();$si.FileName=PYTHON;$si.WorkingDirectory=ROOT;
$si.UseShellExecute=$false;$si.CreateNoWindow=$true;
$si.RedirectStandardInput=$true;$si.RedirectStandardOutput=$true;$si.RedirectStandardError=$true;
foreach($a in @('-u','-m','scripts.qmt_data_bridge','--market-port',PORT,'--timeout',LIMIT)){$si.ArgumentList.Add($a)}
$p=[Diagnostics.Process]::new();$p.StartInfo=$si;[void]$p.Start();
$o=$p.StandardOutput.ReadToEndAsync();$e=$p.StandardError.ReadToEndAsync();
$p.StandardInput.Write([Text.Encoding]::ASCII.GetString([Convert]::FromBase64String(BODY)));$p.StandardInput.Close();
try {
 if(-not $p.WaitForExit(OUTER)){ & taskkill.exe /PID $p.Id /T /F | Out-Null;[void]$p.WaitForExit(3000);throw 'bridge deadline' }
 if(-not $o.Wait(2000)){throw 'bridge output deadline'}
 [Console]::Out.Write($o.Result)
} finally {if(-not $p.HasExited){& taskkill.exe /PID $p.Id /T /F | Out-Null};$p.Dispose()}
"""
    substitutions = {"PYTHON": _ps(py), "ROOT": _ps(root), "PORT": _ps(str(port)),
                     "LIMIT": _ps(str(timeout)), "OUTER": str(int((timeout + 5) * 1000)),
                     "BODY": _ps(base64.b64encode(canonical_bytes(request)).decode("ascii"))}
    script = re.sub(r"\b(PYTHON|ROOT|PORT|LIMIT|OUTER|BODY)\b", lambda m: substitutions[m[0]], script)
    command = '"' + pwsh + '" -NoLogo -NoProfile -NonInteractive -EncodedCommand ' + base64.b64encode(script.encode("utf-16le")).decode("ascii")
    try:
        result = subprocess.run(args + [destination, command], capture_output=True, timeout=timeout + 18)
        if result.returncode or len(result.stdout) > 8 * 1024 * 1024:
            raise MiniQMTError("MiniQMT SSH bridge failed or exceeded its reply bound")
        return json.loads(result.stdout)
    except (subprocess.SubprocessError, OSError, ValueError):
        # Never include SSH arguments, remote paths, SDK diagnostics or private
        # configuration contents in public errors/logs.
        raise MiniQMTError("MiniQMT SSH bridge is unavailable or timed out") from None


class MiniQMTBridgeClient:
    def __init__(self, *, transport: Callable | None = None, timeout: float | None = None):
        if transport is None:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        self.timeout = float(timeout if timeout is not None else os.getenv("MINIQMT_TIMEOUT", "30"))
        if not 0 < self.timeout <= 120:
            raise ValueError("MiniQMT total timeout must be in (0, 120]")
        self._transport = transport

    @staticmethod
    def configured() -> bool:
        if os.getenv("MINIQMT_ENABLED", "true").lower() in {"0", "false", "no", "off"}:
            return False
        root = _private_directory()
        return (root / "machine_configuration.yaml").is_file() and (root / "connection-targets.env").is_file()

    def request(self, operation, symbols, start, end, *, adjustment="none") -> dict:
        begin, finish = _day(start), _day(end)
        if (operation not in {"bars", "fetch_bars", "update_pair", "calendar", "factors", "quotes"}
                or not isinstance(symbols, list) or not 1 <= len(symbols) <= 20
                or len(set(symbols)) != len(symbols)
                or not all(isinstance(s, str) and re.fullmatch(r"SH|SZ" if operation == "calendar" else r"\d{6}\.(SH|SZ)", s) for s in symbols)
                or not 0 <= (finish - begin).days <= (400 if operation in {"bars", "fetch_bars", "update_pair"} else 365 * 30)
                or adjustment not in {"none", "front", "back", "front_ratio", "back_ratio"}
                or (operation not in {"bars"} and adjustment != "none")):
            raise MiniQMTError("invalid bounded market request")
        request = {"operation": operation, "symbols": list(symbols), "start": begin.strftime("%Y%m%d"),
                   "end": finish.strftime("%Y%m%d"), "adjustment": adjustment}
        result = self._transport(request) if self._transport else _ssh_request(request, self.timeout)
        try:
            if isinstance(result, dict) and result.get("status") != "ok":
                detail = result.get("phase") or result.get("error_type") or "bridge"
                if not isinstance(detail, str) or not re.fullmatch(r"[A-Za-z_]+", detail):
                    detail = "bridge"
                raise MiniQMTError(f"MiniQMT request {str(result.get('status'))[:40]} at {detail}")
            if (not isinstance(result, dict) or result.get("status") != "ok" or result.get("provider") != "miniqmt"
                    or result.get("request") != request
                    or result.get("request_sha256") != hashlib.sha256(canonical_bytes(request)).hexdigest()):
                raise MiniQMTError("MiniQMT request failed or returned a different identity")
            if result.get("data_sha256") != hashlib.sha256(canonical_bytes(result["data"])).hexdigest():
                raise MiniQMTError("MiniQMT response digest differs")
            if set(result["data"]) != set(symbols):
                raise MiniQMTError("MiniQMT returned a different instrument scope")
            return result
        except (KeyError, TypeError, ValueError):
            raise MiniQMTError("MiniQMT response is malformed") from None

    def daily(self, symbols, start, end, *, adjustment="none"):
        """Share the bridge's single cache/missing-date decision for raw and front."""
        if adjustment not in {"none", "front"}:
            raise MiniQMTError("daily provider supports raw and front prices")
        begin, finish = _day(start), _day(end)
        outputs = {s: [] for s in symbols}
        receipts = []
        while begin <= finish:
            until = min(finish, begin + timedelta(days=400))
            response = self.request("update_pair", symbols, begin, until)
            receipts.append(response)
            for symbol in symbols:
                raw, qfq = verified_pair(response["data"][symbol], begin, until)
                selected = qfq if adjustment == "front" else raw
                outputs[symbol].extend(selected.to_dict("records"))
            begin = until + timedelta(days=1)
        return {s: pd.DataFrame(v) for s, v in outputs.items()}, receipts


def adjusted_prices(raw: pd.DataFrame, actions: list[dict]) -> pd.DataFrame:
    """Cash/share-event affine front adjustment; native ``dr`` is not a ratio."""
    result = raw.copy()
    days = pd.to_datetime(result["date"]).dt.normalize()
    for field in ("open", "high", "low", "close"):
        result[field] = pd.to_numeric(result[field], errors="raise").astype(float)
    seen = set()
    for event in sorted(actions, key=lambda e: e["date"]):
        day = _day(event["date"])
        if day in seen:
            raise MiniQMTError("duplicate MiniQMT action date")
        seen.add(day)
        values = {}
        for field in ("interest", "stockBonus", "stockGift", "allotNum", "allotPrice", "gugai"):
            value = event.get(field)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise MiniQMTError("incomplete MiniQMT cash/share action")
            values[field] = value
        ratio = 1 + values["stockBonus"] + values["stockGift"] + values["allotNum"]
        cash = values["interest"] - values["allotNum"] * values["allotPrice"]
        if ratio <= 0 or values["gugai"] != 0:
            raise MiniQMTError("unsupported MiniQMT share conversion")
        before = days < pd.Timestamp(day)
        for field in ("open", "high", "low", "close"):
            result.loc[before, field] = (pd.to_numeric(result.loc[before, field]) - cash) / ratio
    return result


def action_history_digest(actions, through, since=None):
    fields = ("interest", "stockBonus", "stockGift", "allotNum", "allotPrice", "gugai")
    records = [dict(date=a["date"], **{k: float(a[k]) for k in fields})
               for a in sorted(actions, key=lambda a: a["date"])
               if _day(a["date"]) <= _day(through) and (since is None or _day(a["date"]) > _day(since))]
    return hashlib.sha256(canonical_bytes(records)).hexdigest()


def verified_pair(data, start, end):
    raw, qfq = pd.DataFrame(data["raw"]), pd.DataFrame(data["qfq"])
    expected_days = set(data["calendar"])
    if raw.empty and qfq.empty and not expected_days:
        return raw, qfq
    if (raw.empty or qfq.empty or len(raw) > 401 or len(qfq) > 401
            or raw["date"].tolist() != qfq["date"].tolist()
            or raw["date"].duplicated().any()
            or set(raw["date"]) != expected_days
            or any(not _day(start) <= _day(d) <= _day(end) for d in expected_days)):
        raise MiniQMTError("MiniQMT pair has unresolved missing, duplicate or closed-day sessions")
    if not (raw["suspendFlag"].to_numpy() == qfq["suspendFlag"].to_numpy()).all():
        raise MiniQMTError("MiniQMT raw/QFQ suspension evidence differs")
    trading = raw["suspendFlag"] == 0
    expected = adjusted_prices(raw.loc[trading], data["actions"])
    fields = ["open", "high", "low", "close", "volume", "amount"]
    if not np.allclose(expected[fields], qfq.loc[trading, fields], rtol=1e-8, atol=1e-8, equal_nan=False):
        raise MiniQMTError("MiniQMT cash/share actions disagree with QFQ prices or volume")
    return raw, qfq
