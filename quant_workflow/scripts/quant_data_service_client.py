"""Fail-closed client for the local governed quant_data read service."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc


DEFAULT_QUANT_DATA_SERVICE_URL = "http://127.0.0.1:8766"
IDENTITY_FIELDS = (
    "dataset_id",
    "snapshot_id",
    "manifest_sha256",
    "content_sha256",
    "latest_trade_date",
    "price_basis",
    "data_kind",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class QuantDataClientError(RuntimeError):
    """A safe error returned by or while contacting the data service."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "service_unavailable",
        status_code: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.details = dict(details or {})


class QuantDataIdentityError(QuantDataClientError):
    """The service omitted or returned malformed immutable identity metadata."""


class QuantDataSnapshotChanged(QuantDataIdentityError):
    """A later response no longer matches the identity locked by the caller."""


@dataclass(frozen=True, slots=True)
class QuantDataIdentity:
    dataset_id: str
    snapshot_id: str
    manifest_sha256: str
    content_sha256: str
    latest_trade_date: str
    price_basis: str
    data_kind: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "QuantDataIdentity":
        if not isinstance(value, Mapping):
            raise QuantDataIdentityError("quant_data response metadata is unavailable")
        selected = {field: value.get(field) for field in IDENTITY_FIELDS}
        if any(not isinstance(item, str) or not item for item in selected.values()):
            raise QuantDataIdentityError("quant_data response metadata is incomplete")
        if not _SHA256.fullmatch(str(selected["manifest_sha256"])) or not _SHA256.fullmatch(
            str(selected["content_sha256"])
        ):
            raise QuantDataIdentityError("quant_data response metadata hashes are invalid")
        return cls(**selected)

    def as_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in IDENTITY_FIELDS}


@dataclass(frozen=True, slots=True)
class QuantDataFrameResult:
    frame: pd.DataFrame
    identity: QuantDataIdentity
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class QuantDataArtifactResult:
    content: bytes
    identity: QuantDataIdentity
    metadata: Mapping[str, Any]
    media_type: str


@dataclass(frozen=True, slots=True)
class QuantDataDatasetResult:
    identity: QuantDataIdentity
    metadata: Mapping[str, Any]
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class QuantDataReadyResult:
    status: str
    validated_datasets: int
    identities: tuple[QuantDataIdentity, ...]
    datasets: tuple[Mapping[str, Any], ...]
    native_release: NativeReleaseIdentity | None = None


@dataclass(frozen=True, slots=True)
class NativeReleaseIdentity:
    generation: int
    pointer_sha256: str
    release_set_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NativeReleaseIdentity":
        generation = value.get("generation")
        pointer = value.get("pointer_sha256")
        release_set = value.get("release_set_sha256")
        if (
            value.get("authority") != "active"
            or value.get("dataset_set_version") != "production.v2"
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 1
            or not isinstance(pointer, str)
            or not _SHA256.fullmatch(pointer)
            or not isinstance(release_set, str)
            or not _SHA256.fullmatch(release_set)
        ):
            raise QuantDataIdentityError("quant_data production.v2 release identity is invalid")
        return cls(generation, pointer, release_set)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "pointer_sha256": self.pointer_sha256,
            "release_set_sha256": self.release_set_sha256,
        }


@dataclass(frozen=True, slots=True)
class NativeDatasetIdentity:
    dataset_id: str
    release: NativeReleaseIdentity
    module_release_sha256: str
    content_manifest_sha256: str
    requested_through: str
    component_identities: Mapping[str, str]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NativeDatasetIdentity":
        release = NativeReleaseIdentity.from_mapping(value)
        module_sha = value.get("module_release_sha256")
        manifest_sha = value.get("content_manifest_sha256")
        components = value.get("component_identities")
        if (
            not isinstance(value.get("dataset_id"), str)
            or value.get("dataset_id") != value.get("module_id")
            or not isinstance(module_sha, str)
            or not _SHA256.fullmatch(module_sha)
            or not isinstance(manifest_sha, str)
            or not _SHA256.fullmatch(manifest_sha)
            or not isinstance(value.get("requested_through"), str)
            or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value["requested_through"])
            or not isinstance(components, Mapping)
            or any(
                not isinstance(key, str)
                or not isinstance(digest, str)
                or not _SHA256.fullmatch(digest)
                for key, digest in components.items()
            )
        ):
            raise QuantDataIdentityError("quant_data production.v2 module identity is invalid")
        return cls(
            str(value["dataset_id"]), release, module_sha, manifest_sha,
            str(value["requested_through"]), dict(components),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            **self.release.as_dict(),
            "module_release_sha256": self.module_release_sha256,
            "content_manifest_sha256": self.content_manifest_sha256,
            "requested_through": self.requested_through,
            "component_identities": dict(self.component_identities),
        }


@dataclass(frozen=True, slots=True)
class NativeDatasetResult:
    identity: NativeDatasetIdentity
    symbols: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NativeFrameResult:
    frame: pd.DataFrame
    identity: NativeDatasetIdentity
    metadata: Mapping[str, Any]


class QuantDataServiceClient:
    """Read only through the loopback quant_data service and preserve lineage."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_QUANT_DATA_SERVICE_URL,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parsed = httpx.URL(str(base_url))
        if (
            parsed.scheme != "http"
            or parsed.host not in _LOCAL_HOSTS
            or bool(parsed.username)
            or bool(parsed.password)
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("local quant_data service URL must be an uncredentialed loopback HTTP origin")
        self.base_url = str(parsed).rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=float(timeout),
            transport=transport,
            follow_redirects=False,
            trust_env=False,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "QuantDataServiceClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise QuantDataClientError("quant_data service is unavailable") from exc
        if response.status_code < 400:
            return response
        code = "service_error"
        message = "quant_data service rejected the request"
        details: Mapping[str, Any] | None = None
        try:
            payload = response.json()
            error = payload.get("error") if isinstance(payload, Mapping) else None
            if isinstance(error, Mapping):
                if isinstance(error.get("code"), str) and error["code"]:
                    code = str(error["code"])
                if isinstance(error.get("message"), str) and error["message"]:
                    message = str(error["message"])
                if isinstance(error.get("details"), Mapping):
                    details = error["details"]
        except (ValueError, json.JSONDecodeError):
            pass
        raise QuantDataClientError(
            message, code=code, status_code=response.status_code, details=details
        )

    @staticmethod
    def _json_mapping(response: httpx.Response, *, label: str) -> Mapping[str, Any]:
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise QuantDataClientError(
                f"quant_data {label} response is invalid", code="invalid_response"
            ) from exc
        if not isinstance(payload, Mapping):
            raise QuantDataClientError(
                f"quant_data {label} response is invalid", code="invalid_response"
            )
        return payload

    @staticmethod
    def _identity(
        metadata: Mapping[str, Any],
        *,
        dataset_id: str,
        expected_identity: QuantDataIdentity | None,
    ) -> QuantDataIdentity:
        identity = QuantDataIdentity.from_mapping(metadata)
        if identity.dataset_id != dataset_id:
            raise QuantDataIdentityError("quant_data response metadata dataset does not match the request")
        if expected_identity is not None and identity != expected_identity:
            raise QuantDataSnapshotChanged(
                f"quant_data snapshot changed while reading {dataset_id}",
                code="snapshot_changed",
                status_code=503,
            )
        return identity

    @staticmethod
    def _header_metadata(response: httpx.Response) -> Mapping[str, Any]:
        serialized = response.headers.get("X-Quant-Data-Metadata")
        if not serialized:
            raise QuantDataIdentityError("quant_data response metadata header is unavailable")
        try:
            metadata = json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise QuantDataIdentityError("quant_data response metadata header is invalid") from exc
        if not isinstance(metadata, Mapping):
            raise QuantDataIdentityError("quant_data response metadata header is invalid")
        return metadata

    @staticmethod
    def _arrow_frame(response: httpx.Response) -> pd.DataFrame:
        try:
            table = ipc.open_stream(pa.BufferReader(response.content)).read_all()
        except (pa.ArrowException, OSError, ValueError) as exc:
            raise QuantDataClientError(
                "quant_data Arrow response is invalid", code="invalid_response"
            ) from exc
        return table.to_pandas()

    def native_ready(self) -> NativeReleaseIdentity:
        payload = self._json_mapping(self._request("GET", "/ready"), label="readiness")
        release = NativeReleaseIdentity.from_mapping(payload)
        datasets = payload.get("datasets")
        validated = payload.get("validated_datasets")
        if (
            payload.get("status") != "ready"
            or isinstance(validated, bool)
            or not isinstance(validated, int)
            or not isinstance(datasets, list)
            or validated != len(datasets)
            or not datasets
            or any(
                not isinstance(item, Mapping)
                or item.get("status") != "ready"
                or item.get("availability") != "AVAILABLE"
                or item.get("dataset_id") != item.get("module_id")
                for item in datasets
            )
        ):
            raise QuantDataClientError(
                "quant_data production.v2 readiness response is inconsistent",
                code="invalid_response",
            )
        return release

    def native_dataset(
        self,
        dataset_id: str,
        *,
        expected_release: NativeReleaseIdentity,
    ) -> NativeDatasetResult:
        payload = self._json_mapping(
            self._request("GET", f"/v1/datasets/{dataset_id}"),
            label="dataset description",
        )
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise QuantDataIdentityError("quant_data module metadata is missing")
        identity = NativeDatasetIdentity.from_mapping(metadata)
        if identity.dataset_id != dataset_id or identity.release != expected_release:
            raise QuantDataSnapshotChanged(
                f"quant_data release changed while describing {dataset_id}",
                code="snapshot_changed",
                status_code=503,
            )
        coverage = payload.get("coverage")
        symbols = coverage.get("symbols", []) if isinstance(coverage, Mapping) else None
        availability = metadata.get("availability")
        if (
            not isinstance(availability, Mapping)
            or availability.get(dataset_id) != "AVAILABLE"
            or not isinstance(symbols, list)
            or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
        ):
            raise QuantDataClientError(
                f"quant_data {dataset_id} is not available for market queries",
                code="invalid_response",
            )
        return NativeDatasetResult(identity, tuple(symbols))

    def native_bars_batch(
        self,
        dataset: NativeDatasetIdentity,
        symbols: Sequence[str],
        *,
        basis: str,
        fields: Sequence[str],
        start: str,
        end: str,
    ) -> NativeFrameResult:
        if not symbols or basis not in dataset.component_identities:
            raise ValueError("native bar request requires symbols and a published basis")
        response = self._request(
            "POST",
            "/v1/bars/query",
            headers={"Accept": "application/vnd.apache.arrow.stream"},
            json={
                "dataset_id": dataset.dataset_id,
                "symbols": list(symbols),
                "basis": basis,
                "fields": list(fields),
                "start": start,
                "end": end,
                "requested_through": end,
                "mode": "strict",
            },
        )
        metadata = self._header_metadata(response)
        actual = NativeDatasetIdentity.from_mapping(metadata)
        staleness = metadata.get("staleness")
        summary = staleness.get("summary") if isinstance(staleness, Mapping) else None
        if actual != dataset or metadata.get("basis") != basis:
            raise QuantDataSnapshotChanged(
                f"quant_data release changed while reading {dataset.dataset_id}",
                code="snapshot_changed",
                status_code=503,
            )
        if (
            not isinstance(staleness, Mapping)
            or staleness.get("mode") != "strict"
            or staleness.get("complete") is not True
            or staleness.get("uncovered") != []
            or staleness.get("requested_through") != end
            or not isinstance(summary, Mapping)
            or summary.get("included_symbol_count") != len(set(symbols))
            or summary.get("excluded_symbol_count") != 0
        ):
            raise QuantDataClientError(
                "quant_data strict query coverage is incomplete",
                code="coverage_unavailable",
            )
        frame = self._arrow_frame(response)
        if not {"symbol", "date", *fields}.issubset(frame.columns):
            raise QuantDataClientError(
                "quant_data bar columns are incomplete", code="invalid_response"
            )
        return NativeFrameResult(frame, actual, dict(metadata))

    def native_named_component(
        self,
        dataset: NativeDatasetIdentity,
        component: str,
        *,
        start: str,
        end: str,
        limit: int = 1000,
    ) -> NativeFrameResult:
        response = self._request(
            "GET",
            f"/v1/artifacts/{dataset.dataset_id}/{component}",
            params={"start": start, "end": end, "limit": limit},
        )
        payload = self._json_mapping(response, label="named component")
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise QuantDataIdentityError("quant_data named component identity is missing")
        actual = NativeDatasetIdentity.from_mapping(metadata)
        if actual != dataset:
            raise QuantDataSnapshotChanged(
                f"quant_data release changed while reading {dataset.dataset_id}",
                code="snapshot_changed",
                status_code=503,
            )
        if (
            payload.get("truncated") is not False
            or not isinstance(payload.get("data"), list)
            or not isinstance(payload.get("columns"), list)
            or payload.get("total_count") != len(payload["data"])
        ):
            raise QuantDataClientError(
                "quant_data named component is incomplete",
                code="coverage_unavailable",
            )
        return NativeFrameResult(
            pd.DataFrame(payload["data"], columns=payload["columns"]),
            actual,
            dict(metadata),
        )

    def ready(self) -> QuantDataReadyResult:
        payload = self._json_mapping(
            self._request("GET", "/ready"), label="readiness"
        )
        if payload.get("dataset_set_version") == "production.v2":
            release = NativeReleaseIdentity.from_mapping(payload)
            datasets = payload.get("datasets")
            validated = payload.get("validated_datasets")
            if (
                payload.get("status") != "ready"
                or isinstance(validated, bool)
                or not isinstance(validated, int)
                or not isinstance(datasets, list)
                or validated != len(datasets)
                or not datasets
            ):
                raise QuantDataClientError(
                    "quant_data production.v2 readiness response is inconsistent",
                    code="invalid_response",
                )
            entries: list[Mapping[str, Any]] = []
            for item in datasets:
                if not isinstance(item, Mapping):
                    raise QuantDataIdentityError("quant_data module identity is invalid")
                identity = NativeDatasetIdentity.from_mapping({**payload, **item})
                if item.get("status") != "ready" or item.get("availability") != "AVAILABLE":
                    raise QuantDataClientError(
                        "quant_data production.v2 readiness response is inconsistent",
                        code="invalid_response",
                    )
                entries.append({
                    "dataset_id": identity.dataset_id,
                    "module_id": identity.dataset_id,
                    "status": "ready",
                    "availability": "AVAILABLE",
                    "requested_through": identity.requested_through,
                    "module_release_sha256": identity.module_release_sha256,
                    "content_manifest_sha256": identity.content_manifest_sha256,
                    "components": list(item.get("components") or []),
                })
            if len({entry["dataset_id"] for entry in entries}) != len(entries):
                raise QuantDataIdentityError("quant_data module identities are duplicated")
            return QuantDataReadyResult(
                status="ready", validated_datasets=validated,
                identities=(), datasets=tuple(entries), native_release=release,
            )
        datasets = payload.get("datasets")
        validated = payload.get("validated_datasets")
        if (
            payload.get("status") != "ready"
            or isinstance(validated, bool)
            or not isinstance(validated, int)
            or not isinstance(datasets, list)
            or validated != len(datasets)
        ):
            raise QuantDataClientError(
                "quant_data readiness response is inconsistent",
                code="invalid_response",
            )
        identities: list[QuantDataIdentity] = []
        entries: list[Mapping[str, Any]] = []
        for dataset in datasets:
            if (
                not isinstance(dataset, Mapping)
                or dataset.get("role") != "canonical"
                or dataset.get("status") != "ready"
            ):
                raise QuantDataClientError(
                    "quant_data readiness response is inconsistent",
                    code="invalid_response",
                )
            identities.append(QuantDataIdentity.from_mapping(dataset))
            entries.append(dict(dataset))
        if len({identity.dataset_id for identity in identities}) != len(identities):
            raise QuantDataClientError(
                "quant_data readiness response is inconsistent",
                code="invalid_response",
            )
        return QuantDataReadyResult(
            status="ready",
            validated_datasets=validated,
            identities=tuple(identities),
            datasets=tuple(entries),
        )

    def describe_dataset(
        self,
        dataset_id: str,
        *,
        expected_identity: QuantDataIdentity | None = None,
    ) -> QuantDataDatasetResult:
        payload = self._json_mapping(
            self._request("GET", f"/v1/datasets/{dataset_id}"),
            label="dataset description",
        )
        metadata = payload.get("metadata")
        identity = self._identity(
            metadata if isinstance(metadata, Mapping) else {},
            dataset_id=dataset_id,
            expected_identity=expected_identity,
        )
        return QuantDataDatasetResult(identity, dict(metadata), dict(payload))

    def resolve_dataset(
        self,
        dataset_id: str,
        *,
        basis: str | None = None,
        expected_identity: QuantDataIdentity | None = None,
    ) -> QuantDataDatasetResult:
        response = self._request(
            "GET",
            f"/v1/datasets/{dataset_id}/resolve",
            params={"basis": basis} if basis is not None else None,
        )
        payload = self._json_mapping(response, label="dataset resolution")
        metadata = payload.get("metadata")
        if (
            payload.get("status") != "ready"
            or not isinstance(metadata, Mapping)
            or metadata.get("validated") is not True
        ):
            raise QuantDataClientError(
                "quant_data dataset resolution response is inconsistent",
                code="invalid_response",
            )
        identity = self._identity(
            metadata,
            dataset_id=dataset_id,
            expected_identity=expected_identity,
        )
        return QuantDataDatasetResult(identity, dict(metadata), dict(payload))

    def query_bars(
        self,
        dataset_id: str,
        symbol: str,
        *,
        basis: str | None = None,
        fields: Sequence[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        expected_identity: QuantDataIdentity | None = None,
    ) -> QuantDataFrameResult:
        params = {
            key: value
            for key, value in {
                "basis": basis,
                "fields": ",".join(fields) if fields is not None else None,
                "start": start,
                "end": end,
            }.items()
            if value is not None
        }
        response = self._request(
            "GET", f"/v1/bars/{dataset_id}/{symbol}", params=params
        )
        payload = self._json_mapping(response, label="JSON")
        metadata = payload.get("metadata")
        identity = self._identity(
            metadata if isinstance(metadata, Mapping) else {},
            dataset_id=dataset_id,
            expected_identity=expected_identity,
        )
        columns = payload.get("columns")
        rows = payload.get("data")
        if not isinstance(columns, list) or not isinstance(rows, list):
            raise QuantDataClientError(
                "quant_data JSON response is invalid", code="invalid_response"
            )
        return QuantDataFrameResult(
            pd.DataFrame(rows, columns=columns), identity, dict(metadata)
        )

    def query_bars_batch(
        self,
        dataset_id: str,
        symbols: Sequence[str],
        *,
        basis: str | None = None,
        fields: Sequence[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        expected_identity: QuantDataIdentity | None = None,
    ) -> QuantDataFrameResult:
        response = self._request(
            "POST",
            "/v1/bars/query",
            headers={"Accept": "application/vnd.apache.arrow.stream"},
            json={
                "dataset_id": dataset_id,
                "symbols": list(symbols),
                "basis": basis,
                "fields": list(fields) if fields is not None else None,
                "start": start,
                "end": end,
            },
        )
        metadata = self._header_metadata(response)
        identity = self._identity(
            metadata,
            dataset_id=dataset_id,
            expected_identity=expected_identity,
        )
        return QuantDataFrameResult(self._arrow_frame(response), identity, dict(metadata))

    def get_artifact(
        self,
        dataset_id: str,
        component: str,
        *,
        expected_identity: QuantDataIdentity | None = None,
    ) -> QuantDataArtifactResult:
        response = self._request("GET", f"/v1/artifacts/{dataset_id}/{component}")
        metadata = self._header_metadata(response)
        identity = self._identity(
            metadata,
            dataset_id=dataset_id,
            expected_identity=expected_identity,
        )
        return QuantDataArtifactResult(
            response.content,
            identity,
            dict(metadata),
            response.headers.get("Content-Type", "application/octet-stream").split(";", 1)[0],
        )

    def query_partitioned_artifact(
        self,
        dataset_id: str,
        component: str,
        *,
        start: str,
        end: str,
        fields: Sequence[str] | None = None,
        symbols: Sequence[str] | None = None,
        expected_identity: QuantDataIdentity | None = None,
    ) -> QuantDataFrameResult:
        params = {
            key: value
            for key, value in {
                "start": start,
                "end": end,
                "fields": ",".join(fields) if fields is not None else None,
                "symbols": ",".join(symbols) if symbols is not None else None,
            }.items()
            if value is not None
        }
        response = self._request(
            "GET",
            f"/v1/artifacts/{dataset_id}/{component}",
            headers={"Accept": "application/vnd.apache.arrow.stream"},
            params=params,
        )
        metadata = self._header_metadata(response)
        identity = self._identity(
            metadata,
            dataset_id=dataset_id,
            expected_identity=expected_identity,
        )
        return QuantDataFrameResult(self._arrow_frame(response), identity, dict(metadata))
