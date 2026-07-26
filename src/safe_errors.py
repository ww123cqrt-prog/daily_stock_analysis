"""Sanitize local error details before exposing them through public APIs."""

from __future__ import annotations

import re
from typing import Optional


_POSIX_SENSITIVE_PATH_ROOTS = (
    "Applications",
    "Library",
    "Users",
    "Volumes",
    "app",
    "etc",
    "home",
    "mnt",
    "opt",
    "private",
    "srv",
    "tmp",
    "usr",
    "var",
)
_POSIX_SENSITIVE_PATH_ROOT_PATTERN = "|".join(
    re.escape(root) for root in _POSIX_SENSITIVE_PATH_ROOTS
)
_WINDOWS_SENSITIVE_PATH_PATTERN = (
    r"[A-Za-z]:(?:[\\/][^\s'\"<>]+|[^\s'\"<>\\/]+[\\/][^\s'\"<>]+)"
)
_WINDOWS_QUOTED_SENSITIVE_PATH_PATTERN = (
    r"[A-Za-z]:(?:[\\/][^\r\n'\"]+|[^\r\n'\"\\/]+[\\/][^\r\n'\"]+)"
)
_QUOTED_ABSOLUTE_PATH_RE = re.compile(
    rf"(['\"])(/(?:{_POSIX_SENSITIVE_PATH_ROOT_PATTERN})(?:/[^\r\n'\"]*)?|{_WINDOWS_QUOTED_SENSITIVE_PATH_PATTERN})\1"
)
_POSIX_ABSOLUTE_PATH_RE = re.compile(
    rf"(?<![A-Za-z0-9_:/.-])/(?:{_POSIX_SENSITIVE_PATH_ROOT_PATTERN})(?:/[^\s'\"<>]+)*"
)
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(
    rf"(?<![A-Za-z0-9_]){_WINDOWS_SENSITIVE_PATH_PATTERN}"
)
_WINDOWS_UNC_PATH_RE = re.compile(
    r"(?<![\\])\\\\[^\\\s'\"<>]+\\[^\\\s'\"<>]+(?:\\[^\\\s'\"<>]+)*"
)


def sanitize_public_error(error: Optional[str]) -> Optional[str]:
    if not error:
        return error
    sanitized = _QUOTED_ABSOLUTE_PATH_RE.sub(r"\1[path]\1", str(error))
    sanitized = _WINDOWS_UNC_PATH_RE.sub("[path]", sanitized)
    sanitized = _WINDOWS_ABSOLUTE_PATH_RE.sub("[path]", sanitized)
    return _POSIX_ABSOLUTE_PATH_RE.sub("[path]", sanitized)
