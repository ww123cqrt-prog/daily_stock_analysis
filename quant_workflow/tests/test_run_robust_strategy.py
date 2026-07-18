from __future__ import annotations

import fcntl
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_robust_strategy.py"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "robust_strategy.yaml"


def load_module():
    spec = importlib.util.spec_from_file_location("run_robust_strategy", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def clean_git_state() -> dict[str, object]:
    return {"commit": "a" * 40, "clean": True, "changes": []}


def write_run_scoped_stage_report(
    report_dir: Path,
    run_id: str,
    stage_name: str,
    file_name: str,
    payload: dict,
) -> None:
    output_dir = report_dir / "runs" / run_id / "stage_reports" / stage_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / file_name).write_text(json.dumps(payload), encoding="utf-8")


def write_pipeline_snapshots(module, quant_data: Path) -> dict[str, str]:
    pit_dir = quant_data / "reference" / "pit_csi800"
    (pit_dir / "daily").mkdir(parents=True, exist_ok=True)
    (pit_dir / "MEMBERSHIP_SOURCE.json").write_text(
        json.dumps(
            {
                "requested": "auto",
                "selected": "baostock",
                "fallback_reason": "jqdata_permission_denied",
                "date_range": {
                    "start_date": "2014-01-02",
                    "end_date": "2026-07-15",
                },
                "indices": {
                    "csi300": "000300.XSHG",
                    "csi500": "000905.XSHG",
                },
            }
        ),
        encoding="utf-8",
    )
    (pit_dir / "membership_intervals.parquet").write_bytes(b"membership intervals")
    (pit_dir / "daily" / "2026.parquet").write_bytes(b"membership daily")

    with module.membership_read_lock(pit_dir) as membership:
        membership_hash = membership.fingerprint()

    reference = quant_data / "reference"
    (reference / "REFERENCE_SOURCE.json").write_text(
        json.dumps(
            {
                "requested": "auto",
                "selected": "baostock",
                "fallback_reason": "jqdata_permission_denied",
                "date_range": {
                    "start_date": "2014-01-02",
                    "end_date": "2026-07-15",
                },
                "industry_classification": "CSRC",
                "membership_snapshot_hash": membership_hash,
            }
        ),
        encoding="utf-8",
    )
    for file_name in (
        "security_master.parquet",
        "industry_intervals.parquet",
        "st_status.parquet",
        "suspension_status.parquet",
    ):
        (reference / file_name).write_bytes(file_name.encode("ascii"))

    with module.reference_read_lock(reference) as reference_snapshot:
        reference_hash = reference_snapshot.fingerprint()
    return {
        "membership_snapshot_hash": membership_hash,
        "reference_snapshot_hash": reference_hash,
    }


def test_load_config_rejects_lowered_safety_or_acceptance_contract(
    tmp_path: Path,
) -> None:
    module = load_module()
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["safety"]["automatic_orders"] = True
    unsafe = tmp_path / "unsafe.yaml"
    unsafe.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    try:
        module.load_config(unsafe)
    except ValueError as exc:
        assert "automatic_orders" in str(exc)
    else:
        raise AssertionError("unsafe strategy config was accepted")

    payload["safety"]["automatic_orders"] = False
    payload["data_gates"]["membership_coverage_min"] = 0.5
    weak = tmp_path / "weak.yaml"
    weak.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    try:
        module.load_config(weak)
    except ValueError as exc:
        assert "membership_coverage_min" in str(exc)
    else:
        raise AssertionError("weakened data gate was accepted")


def test_stage_commands_are_isolated_under_robust_namespace(tmp_path: Path) -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)

    stages = module.build_stages(
        config,
        config_path=CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports" / "robust_strategy",
        as_of_date="2026-07-15",
        skip_membership_build=False,
        run_id="resume-me",
        resume=True,
    )

    assert [stage.name for stage in stages] == [
        "build_pit_csi800",
        "download_robust_reference",
        "add_robust_benchmarks",
        "preflight_robust_data",
        "prepare_robust_provider",
        "run_robust_development",
    ]
    benchmark_stage = next(
        stage for stage in stages if stage.name == "add_robust_benchmarks"
    )
    provider_index = benchmark_stage.argv.index("--provider-dir")
    assert benchmark_stage.argv[provider_index + 1] == str(
        tmp_path / "quant_data" / "qlib" / "cn_data_baostock_jqdata_cleaned"
    )
    required_start_index = benchmark_stage.argv.index("--required-start-date")
    assert benchmark_stage.argv[required_start_index + 1] == "2014-01-02"
    provider_stage = next(
        stage for stage in stages if stage.name == "prepare_robust_provider"
    )
    manifest_index = provider_stage.argv.index("--data-manifest")
    assert provider_stage.argv[manifest_index + 1] == str(
        (tmp_path / "quant_data" / "DATA_MANIFEST.json").resolve()
    )
    development_stage = next(
        stage for stage in stages if stage.name == "run_robust_development"
    )
    assert (
        development_stage.argv[development_stage.argv.index("--run-id") + 1]
        == "resume-me"
    )
    stage_report_root = (
        tmp_path
        / "reports"
        / "robust_strategy"
        / "runs"
        / "resume-me"
        / "stage_reports"
    )
    for stage, option in (
        (stages[0], "--status-out-dir"),
        (
            next(item for item in stages if item.name == "download_robust_reference"),
            "--status-out-dir",
        ),
        (
            next(item for item in stages if item.name == "preflight_robust_data"),
            "--out-dir",
        ),
        (provider_stage, "--status-out-dir"),
    ):
        assert Path(stage.argv[stage.argv.index(option) + 1]) == (
            stage_report_root / stage.name
        )
    assert Path(
        development_stage.argv[development_stage.argv.index("--provider-state") + 1]
    ) == (
        tmp_path
        / "reports"
        / "robust_strategy"
        / "runs"
        / "resume-me"
        / "evidence"
        / "prepare_robust_provider.json"
    )
    assert "--resume" in development_stage.argv
    assert all(
        str(tmp_path / "reports" / "robust_strategy") in " ".join(stage.argv)
        for stage in stages
        if stage.name != "add_robust_benchmarks"
    )
    joined = " ".join(part for stage in stages for part in stage.argv)
    membership_stage = stages[0]
    assert (
        membership_stage.argv[membership_stage.argv.index("--membership-source") + 1]
        == "auto"
    )
    reference_stage = next(
        stage for stage in stages if stage.name == "download_robust_reference"
    )
    assert (
        reference_stage.argv[reference_stage.argv.index("--reference-source") + 1]
        == "auto"
    )
    assert (
        reference_stage.argv[reference_stage.argv.index("--run-id") + 1]
        == "resume-me"
    )
    assert "candidate_pipeline_latest" not in joined
    assert "latest_training_result" not in joined


def test_stage_evidence_rejects_shared_latest_from_other_or_unidentified_run(
    tmp_path: Path,
) -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)
    report_dir = tmp_path / "reports" / "robust_strategy"
    report_dir.mkdir(parents=True)
    stage = module.Stage(
        "run_robust_development",
        ["python", "run_robust_development.py", "--run-id", "run-a"],
    )

    rejected = []
    for payload in (
        {"status": "research_candidate_ready", "run_id": "run-b"},
        {"status": "research_candidate_ready"},
    ):
        (report_dir / "development_latest.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        rejected.append(
            module._stage_evidence(stage, report_dir, tmp_path / "quant_data", config)
        )

    assert [item.get("status") for item in rejected] == [
        "blocked_data",
        "blocked_data",
    ]
    assert [item.get("reason_code") for item in rejected] == [
        "stage_evidence_unverified",
        "stage_evidence_unverified",
    ]


def test_resume_rejects_same_run_shared_development_without_full_context(
    tmp_path: Path,
) -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)
    report_dir = tmp_path / "reports" / "robust_strategy"
    report_dir.mkdir(parents=True)
    (report_dir / "development_latest.json").write_text(
        json.dumps({"status": "research_candidate_ready", "run_id": "run-a"}),
        encoding="utf-8",
    )
    development_dir = report_dir / "runs" / "run-a" / "development"
    development_dir.mkdir(parents=True)
    (development_dir / "development_context.json").write_text(
        json.dumps({"context_hash": "development-context"}), encoding="utf-8"
    )
    stage = module.Stage(
        "run_robust_development",
        [
            "python",
            "run_robust_development.py",
            "--run-id",
            "run-a",
            "--resume",
        ],
    )
    state = {
        "run_id": "run-a",
        "config_hash": "config-hash",
        "git_commit": "a" * 40,
        "data_manifest_hash": "manifest-hash",
        "as_of_date": "2026-07-15",
        "membership_snapshot_hash": "membership-hash",
        "reference_snapshot_hash": "reference-hash",
        "provider_uri": str(tmp_path / "runtime-provider"),
        "provider_key": "provider-key",
        "provider_tree_sha256": "b" * 64,
    }

    evidence = module._stage_evidence(
        stage,
        report_dir,
        tmp_path / "quant_data",
        config,
        run_id="run-a",
        state=state,
    )

    assert evidence == {
        "status": "blocked_data",
        "reason_code": "stage_evidence_unverified",
    }
    assert not (
        report_dir / "runs" / "run-a" / "evidence" / "run_robust_development.json"
    ).exists()


def test_preflight_rejects_same_run_shared_latest_with_only_snapshot_identity(
    tmp_path: Path,
) -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)
    report_dir = tmp_path / "reports" / "robust_strategy"
    report_dir.mkdir(parents=True)
    (report_dir / "preflight_latest.json").write_text(
        json.dumps(
            {
                "status": "ready",
                "run_id": "run-a",
                "as_of_date": "2026-07-15",
                "membership_snapshot_hash": "membership-hash",
                "reference_snapshot_hash": "reference-hash",
            }
        ),
        encoding="utf-8",
    )
    stage = module.Stage(
        "preflight_robust_data",
        ["python", "preflight_robust_data.py"],
    )
    state = {
        "run_id": "run-a",
        "config_hash": "config-hash",
        "git_commit": "a" * 40,
        "data_manifest_hash": "manifest-hash",
        "as_of_date": "2026-07-15",
        "membership_snapshot_hash": "membership-hash",
        "reference_snapshot_hash": "reference-hash",
        "provider_uri": str(tmp_path / "runtime-provider"),
        "provider_key": "provider-key",
        "provider_tree_sha256": "b" * 64,
    }

    evidence = module._stage_evidence(
        stage,
        report_dir,
        tmp_path / "quant_data",
        config,
        run_id="run-a",
        state=state,
    )

    assert evidence == {
        "status": "blocked_data",
        "reason_code": "stage_evidence_unverified",
    }
    assert not (
        report_dir / "runs" / "run-a" / "evidence" / "preflight_robust_data.json"
    ).exists()


def test_stage_evidence_rejects_complete_shared_development_report(
    tmp_path: Path,
) -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)
    report_dir = tmp_path / "reports" / "robust_strategy"
    report_dir.mkdir(parents=True)
    development_dir = report_dir / "runs" / "run-a" / "development"
    development_dir.mkdir(parents=True)
    (development_dir / "development_context.json").write_text(
        json.dumps({"context_hash": "development-context"}), encoding="utf-8"
    )
    state = {
        "run_id": "run-a",
        "config_hash": "config-hash",
        "git_commit": "a" * 40,
        "data_manifest_hash": "manifest-hash",
        "as_of_date": "2026-07-15",
        "membership_snapshot_hash": "membership-hash",
        "reference_snapshot_hash": "reference-hash",
        "provider_uri": str(tmp_path / "runtime-provider"),
        "provider_key": "provider-key",
        "provider_tree_sha256": "b" * 64,
    }
    (report_dir / "development_latest.json").write_text(
        json.dumps(
            {
                "status": "research_candidate_ready",
                **state,
                "development_context_hash": "development-context",
            }
        ),
        encoding="utf-8",
    )
    stage = module.Stage(
        "run_robust_development",
        [
            "python",
            "run_robust_development.py",
            "--run-id",
            "run-a",
            "--resume",
        ],
    )

    evidence = module._stage_evidence(
        stage,
        report_dir,
        tmp_path / "quant_data",
        config,
        run_id="run-a",
        state=state,
    )

    assert evidence == {
        "status": "blocked_data",
        "reason_code": "stage_evidence_unverified",
    }
    assert not (
        report_dir / "runs" / "run-a" / "evidence" / "run_robust_development.json"
    ).exists()


def test_resume_rejects_canonicalized_shared_development_without_original_identity(
    tmp_path: Path,
) -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)
    report_dir = tmp_path / "reports" / "robust_strategy"
    evidence_dir = report_dir / "runs" / "run-a" / "evidence"
    evidence_dir.mkdir(parents=True)
    state = {
        "run_id": "run-a",
        "config_hash": "config-hash",
        "git_commit": "a" * 40,
        "data_manifest_hash": "manifest-hash",
        "as_of_date": "2026-07-15",
        "membership_snapshot_hash": "membership-hash",
        "reference_snapshot_hash": "reference-hash",
        "provider_uri": str(tmp_path / "runtime-provider"),
        "provider_key": "provider-key",
        "provider_tree_sha256": "b" * 64,
    }
    pipeline_context = dict(state)
    (evidence_dir / "run_robust_development.json").write_text(
        json.dumps(
            {
                "status": "research_candidate_ready",
                "run_id": "run-a",
                "evidence_stage": "run_robust_development",
                "evidence_source": "shared_latest",
                "pipeline_context": pipeline_context,
                "pipeline_context_hash": module._payload_sha256(pipeline_context),
            }
        ),
        encoding="utf-8",
    )
    stage = module.Stage(
        "run_robust_development",
        [
            "python",
            "run_robust_development.py",
            "--run-id",
            "run-a",
            "--resume",
        ],
    )

    evidence = module._stage_evidence(
        stage,
        report_dir,
        tmp_path / "quant_data",
        config,
        run_id="run-a",
        state=state,
    )

    assert evidence == {
        "status": "blocked_data",
        "reason_code": "stage_evidence_context_mismatch",
    }


def test_stage_evidence_prefers_and_persists_run_scoped_report(tmp_path: Path) -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)
    report_dir = tmp_path / "reports" / "robust_strategy"
    development_dir = report_dir / "runs" / "run-a" / "development"
    development_dir.mkdir(parents=True)
    (development_dir / "development.json").write_text(
        json.dumps({"status": "research_candidate_ready", "run_id": "run-a"}),
        encoding="utf-8",
    )
    (report_dir / "development_latest.json").write_text(
        json.dumps({"status": "development_failed", "run_id": "run-b"}),
        encoding="utf-8",
    )
    stage = module.Stage(
        "run_robust_development",
        ["python", "run_robust_development.py", "--run-id", "run-a"],
    )

    evidence = module._stage_evidence(
        stage, report_dir, tmp_path / "quant_data", config
    )

    assert evidence["status"] == "research_candidate_ready"
    assert evidence["run_id"] == "run-a"
    persisted = json.loads(
        (
            report_dir / "runs" / "run-a" / "evidence" / "run_robust_development.json"
        ).read_text(encoding="utf-8")
    )
    assert persisted["status"] == "research_candidate_ready"
    assert persisted["run_id"] == "run-a"


def test_pipeline_rejects_run_scoped_evidence_with_wrong_config_hash(
    tmp_path: Path,
) -> None:
    module = load_module()
    module.runtime_provider_identity = lambda provider, key: {
        "provider_key": key,
        "provider_tree_sha256": "b" * 64,
    }
    report_dir = tmp_path / "reports" / "robust_strategy"
    quant_data = tmp_path / "quant_data"
    write_pipeline_snapshots(module, quant_data)

    def write_stage_report(stage, option: str, name: str, payload: dict) -> None:
        output_dir = Path(stage.argv[stage.argv.index(option) + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    def runner(stage):
        if stage.name == "download_robust_reference":
            write_stage_report(
                stage,
                "--status-out-dir",
                "download_robust_reference_latest.json",
                {"status": "completed"},
            )
            return 0
        if stage.name == "add_robust_benchmarks":
            return 0
        if stage.name == "preflight_robust_data":
            write_stage_report(
                stage,
                "--out-dir",
                "preflight_latest.json",
                {"status": "ready"},
            )
            return 0
        if stage.name == "prepare_robust_provider":
            write_stage_report(
                stage,
                "--status-out-dir",
                "provider_latest.json",
                {
                    "status": "completed",
                    "provider_uri": str(tmp_path / "runtime-provider"),
                    "provider_key": "provider-key",
                    "provider_tree_sha256": "b" * 64,
                },
            )
            return 0
        if stage.name == "run_robust_development":
            development_dir = report_dir / "runs" / "run-a" / "development"
            development_dir.mkdir(parents=True, exist_ok=True)
            (development_dir / "development.json").write_text(
                json.dumps(
                    {
                        "status": "research_candidate_ready",
                        "run_id": "run-a",
                        "config_hash": "wrong-config-hash",
                        "provider_key": "provider-key",
                        "provider_tree_sha256": "b" * 64,
                    }
                ),
                encoding="utf-8",
            )
            return 0
        raise AssertionError(stage.name)

    result = module.run_pipeline(
        CONFIG_PATH,
        quant_data_dir=quant_data,
        report_dir=report_dir,
        as_of_date="2026-07-15",
        run_id="run-a",
        skip_membership_build=True,
        stage_runner=runner,
        git_state_provider=clean_git_state,
    )

    assert result["status"] == "development_failed"
    assert result["failed_stage"] == "run_robust_development"
    assert result["reason_code"] == "stage_evidence_context_mismatch"


def test_pipeline_stops_at_membership_permission_block_and_writes_latest_status(
    tmp_path: Path,
) -> None:
    module = load_module()
    calls: list[str] = []
    report_dir = tmp_path / "reports" / "robust_strategy"

    def runner(stage):
        calls.append(stage.name)
        if stage.name == "build_pit_csi800":
            status_dir = Path(stage.argv[stage.argv.index("--status-out-dir") + 1])
            status_dir.mkdir(parents=True, exist_ok=True)
            (status_dir / "build_status_latest.json").write_text(
                json.dumps(
                    {
                        "status": "blocked_data",
                        "reason_code": "jqdata_permission_denied",
                    }
                ),
                encoding="utf-8",
            )
            return 2
        raise AssertionError("later stages must not run")

    result = module.run_pipeline(
        CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=report_dir,
        as_of_date="2026-07-15",
        run_id="test-blocked-membership",
        stage_runner=runner,
        git_state_provider=clean_git_state,
    )

    latest = json.loads(
        (report_dir / "robust_strategy_status_latest.json").read_text(encoding="utf-8")
    )
    assert calls == ["build_pit_csi800"]
    assert result["status"] == "blocked_data"
    assert latest["status"] == "blocked_data"
    assert latest["reason_code"] == "jqdata_permission_denied"
    assert not (report_dir.parent / "candidate_pipeline_latest.json").exists()
    assert not (report_dir.parent / "latest_training_result.json").exists()


def test_pipeline_stops_at_failed_preflight_when_membership_build_is_skipped(
    tmp_path: Path,
) -> None:
    module = load_module()
    calls: list[str] = []
    report_dir = tmp_path / "reports" / "robust_strategy"
    quant_data = tmp_path / "quant_data"
    write_pipeline_snapshots(module, quant_data)

    def runner(stage):
        calls.append(stage.name)
        if stage.name == "download_robust_reference":
            status_dir = Path(stage.argv[stage.argv.index("--status-out-dir") + 1])
            status_dir.mkdir(parents=True, exist_ok=True)
            (status_dir / "download_robust_reference_latest.json").write_text(
                json.dumps({"status": "completed", "reason_code": "completed"}),
                encoding="utf-8",
            )
            return 0
        if stage.name == "add_robust_benchmarks":
            return 0
        if stage.name == "preflight_robust_data":
            out_dir = Path(stage.argv[stage.argv.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "preflight_latest.json").write_text(
                json.dumps(
                    {
                        "status": "blocked_data",
                        "summary": {"failed_checks": ["required_benchmarks"]},
                    }
                ),
                encoding="utf-8",
            )
            return 2
        raise AssertionError("development must not run")

    result = module.run_pipeline(
        CONFIG_PATH,
        quant_data_dir=quant_data,
        report_dir=report_dir,
        as_of_date="2026-07-15",
        run_id="test-blocked-preflight",
        skip_membership_build=True,
        stage_runner=runner,
        git_state_provider=clean_git_state,
    )

    assert calls == [
        "download_robust_reference",
        "add_robust_benchmarks",
        "preflight_robust_data",
    ]
    assert result["status"] == "blocked_data"
    assert result["failed_stage"] == "preflight_robust_data"


def test_pipeline_prepares_isolated_provider_only_after_preflight_passes(
    tmp_path: Path,
) -> None:
    module = load_module()
    module.runtime_provider_identity = lambda provider, key: {
        "provider_key": key,
        "provider_tree_sha256": "c" * 64,
    }
    calls: list[str] = []
    report_dir = tmp_path / "reports" / "robust_strategy"
    quant_data = tmp_path / "quant_data"
    hashes = write_pipeline_snapshots(module, quant_data)

    def runner(stage):
        calls.append(stage.name)
        if stage.name == "download_robust_reference":
            status_dir = Path(stage.argv[stage.argv.index("--status-out-dir") + 1])
            status_dir.mkdir(parents=True, exist_ok=True)
            (status_dir / "download_robust_reference_latest.json").write_text(
                json.dumps({"status": "completed", "reason_code": "completed"}),
                encoding="utf-8",
            )
            return 0
        if stage.name == "add_robust_benchmarks":
            return 0
        if stage.name == "preflight_robust_data":
            out_dir = Path(stage.argv[stage.argv.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "preflight_latest.json").write_text(
                json.dumps({"status": "ready", "summary": {"failed_checks": []}}),
                encoding="utf-8",
            )
            return 0
        if stage.name == "prepare_robust_provider":
            status_dir = Path(stage.argv[stage.argv.index("--status-out-dir") + 1])
            status_dir.mkdir(parents=True, exist_ok=True)
            (status_dir / "provider_latest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "provider_uri": str(tmp_path / "runtime-provider"),
                        "provider_key": "provider-key",
                        "provider_tree_sha256": "c" * 64,
                    }
                ),
                encoding="utf-8",
            )
            return 0
        if stage.name == "run_robust_development":
            development_dir = report_dir / "runs" / "test-ready" / "development"
            development_dir.mkdir(parents=True, exist_ok=True)
            (development_dir / "development.json").write_text(
                json.dumps(
                    {"status": "research_candidate_ready", "run_id": "test-ready"}
                ),
                encoding="utf-8",
            )
            return 0
        raise AssertionError(stage.name)

    result = module.run_pipeline(
        CONFIG_PATH,
        quant_data_dir=quant_data,
        report_dir=report_dir,
        as_of_date="2026-07-15",
        run_id="test-ready",
        skip_membership_build=True,
        stage_runner=runner,
        git_state_provider=clean_git_state,
    )

    assert calls == [
        "download_robust_reference",
        "add_robust_benchmarks",
        "preflight_robust_data",
        "prepare_robust_provider",
        "run_robust_development",
    ]
    assert result["status"] == "research_candidate_ready"
    assert result["membership_snapshot_hash"] == hashes["membership_snapshot_hash"]
    assert result["reference_snapshot_hash"] == hashes["reference_snapshot_hash"]


def test_resume_without_status_starts_fresh_and_runs_every_stage(
    tmp_path: Path,
) -> None:
    module = load_module()
    module.runtime_provider_identity = lambda provider, key: {
        "provider_key": key,
        "provider_tree_sha256": "e" * 64,
    }
    report_dir = tmp_path / "reports" / "robust_strategy"
    quant_data = tmp_path / "quant_data"
    write_pipeline_snapshots(module, quant_data)
    calls: list[str] = []
    development_argv: list[str] = []

    def write_report(stage, option: str, name: str, payload: dict) -> None:
        output_dir = Path(stage.argv[stage.argv.index(option) + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    def runner(stage):
        calls.append(stage.name)
        if stage.name == "build_pit_csi800":
            write_report(
                stage,
                "--status-out-dir",
                "build_status_latest.json",
                {"status": "completed"},
            )
        elif stage.name == "download_robust_reference":
            write_report(
                stage,
                "--status-out-dir",
                "download_robust_reference_latest.json",
                {"status": "completed"},
            )
        elif stage.name == "preflight_robust_data":
            write_report(
                stage,
                "--out-dir",
                "preflight_latest.json",
                {"status": "ready"},
            )
        elif stage.name == "prepare_robust_provider":
            write_report(
                stage,
                "--status-out-dir",
                "provider_latest.json",
                {
                    "status": "completed",
                    "provider_uri": str(tmp_path / "runtime-provider"),
                    "provider_key": "provider-key",
                    "provider_tree_sha256": "e" * 64,
                },
            )
        elif stage.name == "run_robust_development":
            development_argv.extend(stage.argv)
            development_dir = report_dir / "runs" / "fresh-resume" / "development"
            development_dir.mkdir(parents=True, exist_ok=True)
            (development_dir / "development.json").write_text(
                json.dumps(
                    {
                        "status": "research_candidate_ready",
                        "run_id": "fresh-resume",
                    }
                ),
                encoding="utf-8",
            )
        return 0

    result = module.run_pipeline(
        CONFIG_PATH,
        quant_data_dir=quant_data,
        report_dir=report_dir,
        as_of_date="2026-07-15",
        run_id="fresh-resume",
        resume=True,
        stage_runner=runner,
        git_state_provider=lambda: {
            "commit": "a" * 40,
            "clean": True,
            "changes": [],
        },
    )

    assert calls == [
        "build_pit_csi800",
        "download_robust_reference",
        "add_robust_benchmarks",
        "preflight_robust_data",
        "prepare_robust_provider",
        "run_robust_development",
    ]
    assert "--resume" in development_argv
    assert result["status"] == "research_candidate_ready"
    assert result["git_clean"] is True


def test_new_run_rejects_dirty_git_before_running_any_stage(tmp_path: Path) -> None:
    module = load_module()
    calls: list[str] = []

    with pytest.raises(ValueError, match="clean git worktree"):
        module.run_pipeline(
            CONFIG_PATH,
            quant_data_dir=tmp_path / "quant_data",
            report_dir=tmp_path / "reports" / "robust_strategy",
            as_of_date="2026-07-15",
            run_id="dirty-new-run",
            stage_runner=lambda stage: calls.append(stage.name) or 0,
            git_state_provider=lambda: {
                "commit": "a" * 40,
                "clean": False,
                "changes": [" M quant_workflow/scripts/run_robust_strategy.py"],
            },
        )

    assert calls == []


def test_resume_rejects_legacy_state_without_git_clean_before_running(
    tmp_path: Path,
) -> None:
    module = load_module()
    report_dir = tmp_path / "reports" / "robust_strategy"
    run_dir = report_dir / "runs" / "legacy-git-state"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "legacy-git-state",
                "status": "blocked_data",
                "config_hash": module.file_sha256(CONFIG_PATH),
                "git_commit": "a" * 40,
                "data_manifest_hash": None,
                "as_of_date": "2026-07-15",
                "stages": {},
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    with pytest.raises(ValueError, match="git_clean"):
        module.run_pipeline(
            CONFIG_PATH,
            quant_data_dir=tmp_path / "quant_data",
            report_dir=report_dir,
            as_of_date="2026-07-15",
            run_id="legacy-git-state",
            resume=True,
            stage_runner=lambda stage: calls.append(stage.name) or 0,
            git_state_provider=lambda: {
                "commit": "a" * 40,
                "clean": True,
                "changes": [],
            },
        )

    assert calls == []


@pytest.mark.parametrize(
    ("target_stage", "invalid_status"),
    [
        ("build_pit_csi800", "ready"),
        ("download_robust_reference", "ready"),
        ("preflight_robust_data", "completed"),
        ("prepare_robust_provider", "ready"),
        ("run_robust_development", "shadow_observation"),
    ],
)
def test_pipeline_rejects_zero_exit_with_invalid_structured_evidence_status(
    tmp_path: Path,
    target_stage: str,
    invalid_status: str,
) -> None:
    module = load_module()
    module.runtime_provider_identity = lambda provider, key: {
        "provider_key": key,
        "provider_tree_sha256": "f" * 64,
    }
    report_dir = tmp_path / "reports" / "robust_strategy"
    quant_data = tmp_path / "quant_data"
    write_pipeline_snapshots(module, quant_data)
    calls: list[str] = []

    def write_report(stage, option: str, name: str, payload: dict) -> None:
        output_dir = Path(stage.argv[stage.argv.index(option) + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    def runner(stage):
        calls.append(stage.name)
        status = invalid_status if stage.name == target_stage else None
        if stage.name == "build_pit_csi800":
            write_report(
                stage,
                "--status-out-dir",
                "build_status_latest.json",
                {"status": status or "completed"},
            )
        elif stage.name == "download_robust_reference":
            write_report(
                stage,
                "--status-out-dir",
                "download_robust_reference_latest.json",
                {"status": status or "completed"},
            )
        elif stage.name == "preflight_robust_data":
            write_report(
                stage,
                "--out-dir",
                "preflight_latest.json",
                {"status": status or "ready"},
            )
        elif stage.name == "prepare_robust_provider":
            write_report(
                stage,
                "--status-out-dir",
                "provider_latest.json",
                {
                    "status": status or "completed",
                    "provider_uri": str(tmp_path / "runtime-provider"),
                    "provider_key": "provider-key",
                    "provider_tree_sha256": "f" * 64,
                },
            )
        elif stage.name == "run_robust_development":
            development_dir = report_dir / "runs" / "invalid-status" / "development"
            development_dir.mkdir(parents=True, exist_ok=True)
            (development_dir / "development.json").write_text(
                json.dumps(
                    {
                        "status": status or "research_candidate_ready",
                        "run_id": "invalid-status",
                    }
                ),
                encoding="utf-8",
            )
        return 0

    result = module.run_pipeline(
        CONFIG_PATH,
        quant_data_dir=quant_data,
        report_dir=report_dir,
        as_of_date="2026-07-15",
        run_id="invalid-status",
        stage_runner=runner,
        git_state_provider=lambda: {
            "commit": "a" * 40,
            "clean": True,
            "changes": [],
        },
    )

    assert result["status"] == (
        "development_failed"
        if target_stage == "run_robust_development"
        else "blocked_data"
    )
    assert result["failed_stage"] == target_stage
    assert result["reason_code"] == "stage_evidence_status_invalid"


@pytest.mark.parametrize(
    "target_stage",
    [
        "build_pit_csi800",
        "download_robust_reference",
        "preflight_robust_data",
        "prepare_robust_provider",
        "run_robust_development",
    ],
)
def test_pipeline_rejects_zero_exit_without_structured_evidence(
    tmp_path: Path,
    target_stage: str,
) -> None:
    module = load_module()
    module.runtime_provider_identity = lambda provider, key: {
        "provider_key": key,
        "provider_tree_sha256": "f" * 64,
    }
    report_dir = tmp_path / "reports" / "robust_strategy"
    quant_data = tmp_path / "quant_data"
    write_pipeline_snapshots(module, quant_data)

    def write_report(stage, option: str, name: str, payload: dict) -> None:
        if stage.name == target_stage:
            return
        output_dir = Path(stage.argv[stage.argv.index(option) + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    def runner(stage):
        if stage.name == "build_pit_csi800":
            write_report(
                stage,
                "--status-out-dir",
                "build_status_latest.json",
                {"status": "completed"},
            )
        elif stage.name == "download_robust_reference":
            write_report(
                stage,
                "--status-out-dir",
                "download_robust_reference_latest.json",
                {"status": "completed"},
            )
        elif stage.name == "preflight_robust_data":
            write_report(
                stage,
                "--out-dir",
                "preflight_latest.json",
                {"status": "ready"},
            )
        elif stage.name == "prepare_robust_provider":
            write_report(
                stage,
                "--status-out-dir",
                "provider_latest.json",
                {
                    "status": "completed",
                    "provider_uri": str(tmp_path / "runtime-provider"),
                    "provider_key": "provider-key",
                    "provider_tree_sha256": "f" * 64,
                },
            )
        elif stage.name == "run_robust_development" and stage.name != target_stage:
            development_dir = report_dir / "runs" / "missing-report" / "development"
            development_dir.mkdir(parents=True, exist_ok=True)
            (development_dir / "development.json").write_text(
                json.dumps(
                    {
                        "status": "research_candidate_ready",
                        "run_id": "missing-report",
                    }
                ),
                encoding="utf-8",
            )
        return 0

    result = module.run_pipeline(
        CONFIG_PATH,
        quant_data_dir=quant_data,
        report_dir=report_dir,
        as_of_date="2026-07-15",
        run_id="missing-report",
        stage_runner=runner,
        git_state_provider=lambda: {
            "commit": "a" * 40,
            "clean": True,
            "changes": [],
        },
    )

    assert result["status"] == (
        "development_failed"
        if target_stage == "run_robust_development"
        else "blocked_data"
    )
    assert result["failed_stage"] == target_stage
    assert result["reason_code"] == "stage_evidence_unverified"


def test_resume_skips_only_stages_recorded_completed(tmp_path: Path) -> None:
    module = load_module()
    report_dir = tmp_path / "reports" / "robust_strategy"
    run_dir = report_dir / "runs" / "resume-me"
    run_dir.mkdir(parents=True)
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    manifest = quant_data / "DATA_MANIFEST.json"
    manifest.write_text('{"version": 1}\n', encoding="utf-8")
    hashes = write_pipeline_snapshots(module, quant_data)
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "resume-me",
                "status": "blocked_data",
                "config_hash": module.file_sha256(CONFIG_PATH),
                "git_commit": "a" * 40,
                "git_clean": True,
                "data_manifest_hash": module.file_sha256(manifest),
                "as_of_date": "2026-07-15",
                **hashes,
                "stages": {
                    "build_pit_csi800": {"status": "completed", "return_code": 0},
                    "download_robust_reference": {
                        "status": "completed",
                        "return_code": 0,
                    },
                    "preflight_robust_data": {
                        "status": "blocked_data",
                        "return_code": 2,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    write_run_scoped_stage_report(
        report_dir,
        "resume-me",
        "build_pit_csi800",
        "build_status_latest.json",
        {"status": "completed"},
    )
    write_run_scoped_stage_report(
        report_dir,
        "resume-me",
        "download_robust_reference",
        "download_robust_reference_latest.json",
        {"status": "completed"},
    )
    calls: list[str] = []

    def runner(stage):
        calls.append(stage.name)
        if stage.name == "add_robust_benchmarks":
            return 0
        if stage.name == "preflight_robust_data":
            out_dir = Path(stage.argv[stage.argv.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "preflight_latest.json").write_text(
                json.dumps(
                    {
                        "status": "blocked_data",
                        "summary": {"failed_checks": ["ohlcv_coverage"]},
                    }
                ),
                encoding="utf-8",
            )
            return 2
        raise AssertionError("development must not run")

    result = module.run_pipeline(
        CONFIG_PATH,
        quant_data_dir=quant_data,
        report_dir=report_dir,
        as_of_date="2026-07-15",
        run_id="resume-me",
        resume=True,
        stage_runner=runner,
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    assert calls == ["add_robust_benchmarks", "preflight_robust_data"]
    assert result["stages"]["build_pit_csi800"]["status"] == "completed"
    assert result["status"] == "blocked_data"


def test_resume_reruns_stage_recorded_completed_with_nonzero_return_code(
    tmp_path: Path,
) -> None:
    module = load_module()
    report_dir = tmp_path / "reports" / "robust_strategy"
    run_dir = report_dir / "runs" / "resume-nonzero"
    run_dir.mkdir(parents=True)
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    manifest = quant_data / "DATA_MANIFEST.json"
    manifest.write_text('{"version": 1}\n', encoding="utf-8")
    hashes = write_pipeline_snapshots(module, quant_data)
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "resume-nonzero",
                "status": "blocked_data",
                "config_hash": module.file_sha256(CONFIG_PATH),
                "git_commit": "a" * 40,
                "git_clean": True,
                "data_manifest_hash": module.file_sha256(manifest),
                "as_of_date": "2026-07-15",
                **hashes,
                "stages": {
                    "build_pit_csi800": {"status": "completed", "return_code": 1}
                },
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def runner(stage):
        calls.append(stage.name)
        assert stage.name == "build_pit_csi800"
        output_dir = Path(stage.argv[stage.argv.index("--status-out-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "build_status_latest.json").write_text(
            json.dumps({"status": "blocked_data"}), encoding="utf-8"
        )
        return 2

    result = module.run_pipeline(
        CONFIG_PATH,
        quant_data_dir=quant_data,
        report_dir=report_dir,
        as_of_date="2026-07-15",
        run_id="resume-nonzero",
        resume=True,
        stage_runner=runner,
        git_state_provider=clean_git_state,
    )

    assert calls == ["build_pit_csi800"]
    assert result["failed_stage"] == "build_pit_csi800"
    assert result["reason_code"] == "stage_evidence_status_invalid"


def test_resume_reruns_completed_preflight_without_run_evidence(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    manifest = quant_data / "DATA_MANIFEST.json"
    manifest.write_text('{"version": 1}\n', encoding="utf-8")
    hashes = write_pipeline_snapshots(module, quant_data)
    report_dir = tmp_path / "reports" / "robust_strategy"
    run_dir = report_dir / "runs" / "missing-preflight-evidence"
    run_dir.mkdir(parents=True)
    write_run_scoped_stage_report(
        report_dir,
        "missing-preflight-evidence",
        "build_pit_csi800",
        "build_status_latest.json",
        {"status": "completed"},
    )
    write_run_scoped_stage_report(
        report_dir,
        "missing-preflight-evidence",
        "download_robust_reference",
        "download_robust_reference_latest.json",
        {"status": "completed"},
    )
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "missing-preflight-evidence",
                "status": "blocked_data",
                "config_hash": module.file_sha256(CONFIG_PATH),
                "git_commit": "a" * 40,
                "git_clean": True,
                "data_manifest_hash": module.file_sha256(manifest),
                "as_of_date": "2026-07-15",
                **hashes,
                "stages": {
                    "build_pit_csi800": {"status": "completed", "return_code": 0},
                    "download_robust_reference": {
                        "status": "completed",
                        "return_code": 0,
                    },
                    "preflight_robust_data": {
                        "status": "completed",
                        "return_code": 0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def runner(stage):
        calls.append(stage.name)
        if stage.name == "add_robust_benchmarks":
            return 0
        if stage.name == "preflight_robust_data":
            out_dir = Path(stage.argv[stage.argv.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "preflight_latest.json").write_text(
                json.dumps(
                    {
                        "status": "blocked_data",
                        "summary": {"failed_checks": ["test_resume_recheck"]},
                    }
                ),
                encoding="utf-8",
            )
            return 2
        return 2

    result = module.run_pipeline(
        CONFIG_PATH,
        quant_data_dir=quant_data,
        report_dir=report_dir,
        as_of_date="2026-07-15",
        run_id="missing-preflight-evidence",
        resume=True,
        stage_runner=runner,
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    assert calls == ["add_robust_benchmarks", "preflight_robust_data"]
    assert result["failed_stage"] == "preflight_robust_data"
    assert result["reason_code"] == "stage_evidence_status_invalid"


def test_resume_materializes_provider_evidence_for_completed_legacy_stage(
    tmp_path: Path,
) -> None:
    module = load_module()
    module.runtime_provider_identity = lambda provider, key: {
        "provider_key": key,
        "provider_tree_sha256": "d" * 64,
    }
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    manifest = quant_data / "DATA_MANIFEST.json"
    manifest.write_text('{"version": 1}\n', encoding="utf-8")
    hashes = write_pipeline_snapshots(module, quant_data)
    report_dir = tmp_path / "reports" / "robust_strategy"
    run_dir = report_dir / "runs" / "legacy-provider"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "legacy-provider",
                "status": "blocked_data",
                "config_hash": module.file_sha256(CONFIG_PATH),
                "git_commit": "a" * 40,
                "git_clean": True,
                "data_manifest_hash": module.file_sha256(manifest),
                "as_of_date": "2026-07-15",
                "provider_uri": str(tmp_path / "runtime-provider"),
                "provider_key": "legacy-key",
                **hashes,
                "stages": {
                    name: {"status": "completed", "return_code": 0}
                    for name in [
                        "build_pit_csi800",
                        "download_robust_reference",
                        "add_robust_benchmarks",
                        "preflight_robust_data",
                        "prepare_robust_provider",
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    write_run_scoped_stage_report(
        report_dir,
        "legacy-provider",
        "build_pit_csi800",
        "build_status_latest.json",
        {"status": "completed"},
    )
    write_run_scoped_stage_report(
        report_dir,
        "legacy-provider",
        "download_robust_reference",
        "download_robust_reference_latest.json",
        {"status": "completed"},
    )
    preflight_dir = run_dir / "stage_reports" / "preflight_robust_data"
    preflight_dir.mkdir(parents=True)
    (preflight_dir / "preflight_latest.json").write_text(
        json.dumps({"status": "ready"}), encoding="utf-8"
    )
    provider_state_seen: list[bool] = []
    calls: list[str] = []

    def runner(stage):
        calls.append(stage.name)
        if stage.name == "add_robust_benchmarks":
            return 0
        assert stage.name == "run_robust_development"
        provider_state = Path(stage.argv[stage.argv.index("--provider-state") + 1])
        provider_state_seen.append(provider_state.is_file())
        development_dir = run_dir / "development"
        development_dir.mkdir(parents=True, exist_ok=True)
        (development_dir / "development.json").write_text(
            json.dumps(
                {
                    "status": "blocked_data",
                    "reason_code": "test_stop",
                    "run_id": "legacy-provider",
                }
            ),
            encoding="utf-8",
        )
        return 2

    result = module.run_pipeline(
        CONFIG_PATH,
        quant_data_dir=quant_data,
        report_dir=report_dir,
        as_of_date="2026-07-15",
        run_id="legacy-provider",
        resume=True,
        stage_runner=runner,
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    assert calls == ["run_robust_development"]
    assert provider_state_seen == [True]
    provider_evidence = json.loads(
        (run_dir / "evidence" / "prepare_robust_provider.json").read_text(
            encoding="utf-8"
        )
    )
    assert provider_evidence["run_id"] == "legacy-provider"
    assert provider_evidence["provider_key"] == "legacy-key"
    assert provider_evidence["provider_tree_sha256"] == "d" * 64
    assert result["status"] == "development_failed"


def test_resume_rejects_changed_membership_before_skipping_any_stage(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    manifest = quant_data / "DATA_MANIFEST.json"
    manifest.write_text('{"version": 1}\n', encoding="utf-8")
    hashes = write_pipeline_snapshots(module, quant_data)
    report_dir = tmp_path / "reports" / "robust_strategy"
    run_dir = report_dir / "runs" / "membership-changed"
    run_dir.mkdir(parents=True)
    state = {
        "run_id": "membership-changed",
        "status": "research_candidate_ready",
        "config_hash": module.file_sha256(CONFIG_PATH),
        "git_commit": "a" * 40,
        "git_clean": True,
        "data_manifest_hash": module.file_sha256(manifest),
        "as_of_date": "2026-07-15",
        **hashes,
        "stages": {
            "build_pit_csi800": {"status": "completed", "return_code": 0},
            "download_robust_reference": {"status": "completed", "return_code": 0},
        },
    }
    (run_dir / "status.json").write_text(json.dumps(state), encoding="utf-8")
    daily = quant_data / "reference" / "pit_csi800" / "daily" / "2026.parquet"
    daily.write_bytes(daily.read_bytes() + b"changed")

    try:
        module.run_pipeline(
            CONFIG_PATH,
            quant_data_dir=quant_data,
            report_dir=report_dir,
            as_of_date="2026-07-15",
            run_id="membership-changed",
            resume=True,
            stage_runner=lambda stage: (_ for _ in ()).throw(
                AssertionError("no stage may be skipped or run")
            ),
            git_state_provider=lambda: {
                "commit": "a" * 40,
                "clean": True,
                "changes": [],
            },
        )
    except ValueError as exc:
        assert "membership snapshot" in str(exc).lower()
    else:
        raise AssertionError("resume accepted changed membership snapshot")


def test_resume_rejects_changed_reference_before_skipping_development(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    manifest = quant_data / "DATA_MANIFEST.json"
    manifest.write_text('{"version": 1}\n', encoding="utf-8")
    hashes = write_pipeline_snapshots(module, quant_data)
    report_dir = tmp_path / "reports" / "robust_strategy"
    run_dir = report_dir / "runs" / "reference-changed"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "reference-changed",
                "status": "research_candidate_ready",
                "config_hash": module.file_sha256(CONFIG_PATH),
                "git_commit": "a" * 40,
                "git_clean": True,
                "data_manifest_hash": module.file_sha256(manifest),
                "as_of_date": "2026-07-15",
                **hashes,
                "stages": {
                    "build_pit_csi800": {"status": "completed", "return_code": 0},
                    "download_robust_reference": {
                        "status": "completed",
                        "return_code": 0,
                    },
                    "run_robust_development": {
                        "status": "completed",
                        "return_code": 0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    reference_file = quant_data / "reference" / "security_master.parquet"
    reference_file.write_bytes(reference_file.read_bytes() + b"changed")

    try:
        module.run_pipeline(
            CONFIG_PATH,
            quant_data_dir=quant_data,
            report_dir=report_dir,
            as_of_date="2026-07-15",
            run_id="reference-changed",
            resume=True,
            stage_runner=lambda stage: (_ for _ in ()).throw(
                AssertionError("no stage may be skipped or run")
            ),
            git_state_provider=lambda: {
                "commit": "a" * 40,
                "clean": True,
                "changes": [],
            },
        )
    except ValueError as exc:
        assert "reference snapshot" in str(exc).lower()
    else:
        raise AssertionError("resume accepted changed reference snapshot")


def test_resume_rejects_reference_stage_without_membership_snapshot_hash(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    manifest = quant_data / "DATA_MANIFEST.json"
    manifest.write_text('{"version": 1}\n', encoding="utf-8")
    hashes = write_pipeline_snapshots(module, quant_data)
    hashes.pop("membership_snapshot_hash")
    report_dir = tmp_path / "reports" / "robust_strategy"
    run_dir = report_dir / "runs" / "missing-membership-hash"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "missing-membership-hash",
                "status": "blocked_data",
                "config_hash": module.file_sha256(CONFIG_PATH),
                "git_commit": "a" * 40,
                "git_clean": True,
                "data_manifest_hash": module.file_sha256(manifest),
                "as_of_date": "2026-07-15",
                **hashes,
                "stages": {
                    "download_robust_reference": {
                        "status": "completed",
                        "return_code": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        module.run_pipeline(
            CONFIG_PATH,
            quant_data_dir=quant_data,
            report_dir=report_dir,
            as_of_date="2026-07-15",
            run_id="missing-membership-hash",
            resume=True,
            stage_runner=lambda stage: (_ for _ in ()).throw(
                AssertionError("no stage may run with incomplete resume identity")
            ),
            git_state_provider=lambda: {
                "commit": "a" * 40,
                "clean": True,
                "changes": [],
            },
        )
    except ValueError as exc:
        assert "membership snapshot hash" in str(exc).lower()
    else:
        raise AssertionError("resume accepted reference state without membership hash")


def test_resume_rejects_completed_provider_with_mismatched_metadata(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    manifest = quant_data / "DATA_MANIFEST.json"
    manifest.write_text('{"version": 1}\n', encoding="utf-8")
    hashes = write_pipeline_snapshots(module, quant_data)
    provider = tmp_path / "runtime-provider"
    (provider / "instruments").mkdir(parents=True)
    (provider / "instruments" / "csi800.txt").write_text("", encoding="utf-8")
    (provider / "ROBUST_PROVIDER.json").write_text(
        json.dumps({"format_version": 2, "provider_key": "different-key"}),
        encoding="utf-8",
    )
    for path in [
        provider / "instruments" / "csi800.txt",
        provider / "ROBUST_PROVIDER.json",
    ]:
        path.chmod(0o444)
    (provider / "instruments").chmod(0o555)
    provider.chmod(0o555)

    report_dir = tmp_path / "reports" / "robust_strategy"
    run_dir = report_dir / "runs" / "provider-changed"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "provider-changed",
                "status": "blocked_data",
                "config_hash": module.file_sha256(CONFIG_PATH),
                "git_commit": "a" * 40,
                "git_clean": True,
                "data_manifest_hash": module.file_sha256(manifest),
                "as_of_date": "2026-07-15",
                "provider_uri": str(provider),
                "provider_key": "expected-key",
                **hashes,
                "stages": {
                    "build_pit_csi800": {"status": "completed", "return_code": 0},
                    "download_robust_reference": {
                        "status": "completed",
                        "return_code": 0,
                    },
                    "add_robust_benchmarks": {"status": "completed", "return_code": 0},
                    "preflight_robust_data": {"status": "completed", "return_code": 0},
                    "prepare_robust_provider": {
                        "status": "completed",
                        "return_code": 0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        module.run_pipeline(
            CONFIG_PATH,
            quant_data_dir=quant_data,
            report_dir=report_dir,
            as_of_date="2026-07-15",
            run_id="provider-changed",
            resume=True,
            stage_runner=lambda stage: (_ for _ in ()).throw(
                AssertionError("no stage may run with a changed provider")
            ),
            git_state_provider=lambda: {
                "commit": "a" * 40,
                "clean": True,
                "changes": [],
            },
        )
    except ValueError as exc:
        assert "provider" in str(exc).lower()
    else:
        raise AssertionError("resume accepted a provider with mismatched metadata")


def test_resume_rejects_changed_manifest_before_skipping_completed_stages(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    manifest = quant_data / "DATA_MANIFEST.json"
    manifest.write_text('{"version": 2}\n', encoding="utf-8")
    old_manifest = tmp_path / "old_manifest.json"
    old_manifest.write_text('{"version": 1}\n', encoding="utf-8")
    report_dir = tmp_path / "reports" / "robust_strategy"
    run_dir = report_dir / "runs" / "resume-changed"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "resume-changed",
                "status": "research_candidate_ready",
                "config_hash": module.file_sha256(CONFIG_PATH),
                "git_commit": "a" * 40,
                "git_clean": True,
                "data_manifest_hash": module.file_sha256(old_manifest),
                "stages": {
                    name: {"status": "completed", "return_code": 0}
                    for name in [
                        "build_pit_csi800",
                        "download_robust_reference",
                        "add_robust_benchmarks",
                        "preflight_robust_data",
                        "prepare_robust_provider",
                        "run_robust_development",
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        module.run_pipeline(
            CONFIG_PATH,
            quant_data_dir=quant_data,
            report_dir=report_dir,
            as_of_date="2026-07-15",
            run_id="resume-changed",
            resume=True,
            stage_runner=lambda stage: (_ for _ in ()).throw(
                AssertionError("no stage may be skipped or run")
            ),
            git_state_provider=lambda: {
                "commit": "a" * 40,
                "clean": True,
                "changes": [],
            },
        )
    except ValueError as exc:
        assert "data manifest" in str(exc).lower()
    else:
        raise AssertionError("resume accepted artifacts from a different data manifest")


def test_config_hash_is_stable_for_same_content(tmp_path: Path) -> None:
    module = load_module()
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    content = CONFIG_PATH.read_text(encoding="utf-8")
    first.write_text(content, encoding="utf-8")
    second.write_text(content, encoding="utf-8")

    assert module.file_sha256(first) == module.file_sha256(second)


def _write_completed_resume_state(
    module,
    *,
    quant_data: Path,
    report_dir: Path,
    run_id: str,
) -> tuple[Path, dict[str, str]]:
    quant_data.mkdir(parents=True, exist_ok=True)
    manifest = quant_data / "DATA_MANIFEST.json"
    manifest.write_text('{"version": 1}\n', encoding="utf-8")
    hashes = write_pipeline_snapshots(module, quant_data)
    run_dir = report_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "run_id": run_id,
        "status": "research_candidate_ready",
        "config_hash": module.file_sha256(CONFIG_PATH),
        "git_commit": "a" * 40,
        "git_clean": True,
        "data_manifest_hash": module.file_sha256(manifest),
        "as_of_date": "2026-07-15",
        "provider_uri": str(run_dir / "runtime-provider"),
        "provider_key": "provider-key",
        "provider_tree_sha256": "d" * 64,
        **hashes,
        "stages": {
            name: {"status": "completed", "return_code": 0}
            for name in [
                "build_pit_csi800",
                "download_robust_reference",
                "add_robust_benchmarks",
                "preflight_robust_data",
                "prepare_robust_provider",
                "run_robust_development",
            ]
        },
    }
    (run_dir / "status.json").write_text(json.dumps(state), encoding="utf-8")
    return run_dir, hashes


def test_resume_reruns_canonical_evidence_without_persisted_identity(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    report_dir = tmp_path / "reports" / "robust_strategy"
    run_id = "legacy-canonical"
    run_dir, _ = _write_completed_resume_state(
        module,
        quant_data=quant_data,
        report_dir=report_dir,
        run_id=run_id,
    )
    module.runtime_provider_identity = lambda provider, key: {
        "provider_key": key,
        "provider_tree_sha256": "d" * 64,
    }
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    statuses = {
        "build_pit_csi800": "completed",
        "download_robust_reference": "completed",
        "preflight_robust_data": "ready",
        "prepare_robust_provider": "completed",
        "run_robust_development": "research_candidate_ready",
    }
    for stage_name, status in statuses.items():
        (evidence_dir / f"{stage_name}.json").write_text(
            json.dumps({"run_id": run_id, "status": status}),
            encoding="utf-8",
        )
    calls: list[str] = []

    def runner(stage):
        calls.append(stage.name)
        return 2

    result = module.run_pipeline(
        CONFIG_PATH,
        quant_data_dir=quant_data,
        report_dir=report_dir,
        as_of_date="2026-07-15",
        run_id=run_id,
        resume=True,
        stage_runner=runner,
        git_state_provider=clean_git_state,
    )

    assert calls == ["build_pit_csi800"]
    assert result["status"] == "blocked_data"


def test_new_run_rejects_existing_run_id_before_overwriting_status(
    tmp_path: Path,
) -> None:
    module = load_module()
    report_dir = tmp_path / "reports" / "robust_strategy"
    status_path = report_dir / "runs" / "duplicate-run" / "status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text('{"sentinel": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        module.run_pipeline(
            CONFIG_PATH,
            quant_data_dir=tmp_path / "quant_data",
            report_dir=report_dir,
            as_of_date="2026-07-15",
            run_id="duplicate-run",
            stage_runner=lambda stage: (_ for _ in ()).throw(AssertionError(stage.name)),
            git_state_provider=clean_git_state,
        )

    assert json.loads(status_path.read_text(encoding="utf-8")) == {"sentinel": True}


def test_run_rejects_concurrent_owner_of_same_run_id(tmp_path: Path) -> None:
    module = load_module()
    report_dir = tmp_path / "reports" / "robust_strategy"
    lock_path = report_dir / "runs" / ".locks" / "same-run.lock"
    lock_path.parent.mkdir(parents=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already running"):
            module.run_pipeline(
                CONFIG_PATH,
                quant_data_dir=tmp_path / "quant_data",
                report_dir=report_dir,
                as_of_date="2026-07-15",
                run_id="same-run",
                resume=True,
                stage_runner=lambda stage: (_ for _ in ()).throw(
                    AssertionError(stage.name)
                ),
                git_state_provider=clean_git_state,
            )


def test_default_run_ids_include_collision_resistant_nonce() -> None:
    module = load_module()

    first = module.new_run_id()
    second = module.new_run_id()

    assert first != second
    assert module.RUN_ID_RE.fullmatch(first)
    assert module.RUN_ID_RE.fullmatch(second)


def test_pipeline_rechecks_git_state_before_publishing_success(tmp_path: Path) -> None:
    module = load_module()
    report_dir = tmp_path / "reports" / "robust_strategy"
    quant_data = tmp_path / "quant_data"
    write_pipeline_snapshots(module, quant_data)
    module.runtime_provider_identity = lambda provider, key: {
        "provider_key": key,
        "provider_tree_sha256": "d" * 64,
    }
    git_states = [
        clean_git_state(),
        {"commit": "a" * 40, "clean": False, "changes": [" M changed.py"]},
    ]

    def runner(stage):
        if stage.name == "download_robust_reference":
            write_run_scoped_stage_report(
                report_dir,
                "final-recheck",
                stage.name,
                "download_robust_reference_latest.json",
                {"status": "completed"},
            )
        elif stage.name == "preflight_robust_data":
            write_run_scoped_stage_report(
                report_dir,
                "final-recheck",
                stage.name,
                "preflight_latest.json",
                {"status": "ready"},
            )
        elif stage.name == "prepare_robust_provider":
            write_run_scoped_stage_report(
                report_dir,
                "final-recheck",
                stage.name,
                "provider_latest.json",
                {
                    "status": "completed",
                    "provider_uri": str(tmp_path / "runtime-provider"),
                    "provider_key": "provider-key",
                    "provider_tree_sha256": "d" * 64,
                },
            )
        elif stage.name == "run_robust_development":
            development_dir = report_dir / "runs" / "final-recheck" / "development"
            development_dir.mkdir(parents=True, exist_ok=True)
            (development_dir / "development.json").write_text(
                json.dumps(
                    {
                        "status": "research_candidate_ready",
                        "run_id": "final-recheck",
                    }
                ),
                encoding="utf-8",
            )
        return 0

    result = module.run_pipeline(
        CONFIG_PATH,
        quant_data_dir=quant_data,
        report_dir=report_dir,
        as_of_date="2026-07-15",
        run_id="final-recheck",
        skip_membership_build=True,
        stage_runner=runner,
        git_state_provider=lambda: git_states.pop(0),
    )

    assert result["status"] == "blocked_data"
    assert result["lifecycle_status"] == "stopped"
    assert result["reason_code"] == "git_worktree_changed"
    assert git_states == []


def test_resume_reruns_provider_when_canonical_evidence_is_corrupt(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    report_dir = tmp_path / "reports" / "robust_strategy"
    run_id = "corrupt-provider"
    run_dir, _ = _write_completed_resume_state(
        module,
        quant_data=quant_data,
        report_dir=report_dir,
        run_id=run_id,
    )
    state = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    state["stages"].pop("run_robust_development")
    (run_dir / "status.json").write_text(json.dumps(state), encoding="utf-8")
    module.runtime_provider_identity = lambda provider, key: {
        "provider_key": key,
        "provider_tree_sha256": "d" * 64,
    }
    write_run_scoped_stage_report(
        report_dir,
        run_id,
        "build_pit_csi800",
        "build_status_latest.json",
        {"status": "completed"},
    )
    write_run_scoped_stage_report(
        report_dir,
        run_id,
        "download_robust_reference",
        "download_robust_reference_latest.json",
        {"status": "completed"},
    )
    write_run_scoped_stage_report(
        report_dir,
        run_id,
        "preflight_robust_data",
        "preflight_latest.json",
        {"status": "ready"},
    )
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    (evidence_dir / "prepare_robust_provider.json").write_text(
        "{truncated", encoding="utf-8"
    )
    calls: list[str] = []

    def runner(stage):
        calls.append(stage.name)
        if stage.name == "prepare_robust_provider":
            write_run_scoped_stage_report(
                report_dir,
                run_id,
                stage.name,
                "provider_latest.json",
                {
                    "status": "completed",
                    "provider_uri": str(run_dir / "runtime-provider"),
                    "provider_key": "provider-key",
                    "provider_tree_sha256": "d" * 64,
                },
            )
            return 0
        development_dir = run_dir / "development"
        development_dir.mkdir(parents=True, exist_ok=True)
        (development_dir / "development.json").write_text(
            json.dumps(
                {
                    "status": "development_failed",
                    "reason_code": "test_stop_after_provider_rebuild",
                    "run_id": run_id,
                }
            ),
            encoding="utf-8",
        )
        return 2

    result = module.run_pipeline(
        CONFIG_PATH,
        quant_data_dir=quant_data,
        report_dir=report_dir,
        as_of_date="2026-07-15",
        run_id=run_id,
        resume=True,
        stage_runner=runner,
        git_state_provider=clean_git_state,
    )

    assert calls == ["prepare_robust_provider", "run_robust_development"]
    assert result["status"] == "development_failed"
    assert result["reason_code"] == "test_stop_after_provider_rebuild"


def test_development_failed_evidence_preserves_original_reason(tmp_path: Path) -> None:
    module = load_module()
    report_dir = tmp_path / "reports" / "robust_strategy"
    quant_data = tmp_path / "quant_data"
    write_pipeline_snapshots(module, quant_data)
    module.runtime_provider_identity = lambda provider, key: {
        "provider_key": key,
        "provider_tree_sha256": "d" * 64,
    }

    def runner(stage):
        if stage.name == "download_robust_reference":
            write_run_scoped_stage_report(
                report_dir,
                "valid-failure",
                stage.name,
                "download_robust_reference_latest.json",
                {"status": "completed"},
            )
        elif stage.name == "preflight_robust_data":
            write_run_scoped_stage_report(
                report_dir,
                "valid-failure",
                stage.name,
                "preflight_latest.json",
                {"status": "ready"},
            )
        elif stage.name == "prepare_robust_provider":
            write_run_scoped_stage_report(
                report_dir,
                "valid-failure",
                stage.name,
                "provider_latest.json",
                {
                    "status": "completed",
                    "provider_uri": str(tmp_path / "runtime-provider"),
                    "provider_key": "provider-key",
                    "provider_tree_sha256": "d" * 64,
                },
            )
        elif stage.name == "run_robust_development":
            development_dir = report_dir / "runs" / "valid-failure" / "development"
            development_dir.mkdir(parents=True, exist_ok=True)
            (development_dir / "development.json").write_text(
                json.dumps(
                    {
                        "status": "development_failed",
                        "reason_code": "acceptance_not_met",
                        "run_id": "valid-failure",
                    }
                ),
                encoding="utf-8",
            )
            return 2
        return 0

    result = module.run_pipeline(
        CONFIG_PATH,
        quant_data_dir=quant_data,
        report_dir=report_dir,
        as_of_date="2026-07-15",
        run_id="valid-failure",
        skip_membership_build=True,
        stage_runner=runner,
        git_state_provider=clean_git_state,
    )

    stage = result["stages"]["run_robust_development"]
    assert stage["status"] == "development_failed"
    assert stage["evidence"]["reason_code"] == "acceptance_not_met"
    assert result["status"] == "development_failed"
    assert result["reason_code"] == "acceptance_not_met"


def test_run_lock_is_held_during_stage_and_released_after_exception(
    tmp_path: Path,
) -> None:
    module = load_module()
    report_dir = tmp_path / "reports" / "robust_strategy"
    lock_path = report_dir / "runs" / ".locks" / "lock-lifetime.lock"
    observed_locked = []

    def runner(stage):
        with lock_path.open("a+", encoding="utf-8") as stream:
            with pytest.raises(BlockingIOError):
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            observed_locked.append(True)
        raise RuntimeError("stop after lock check")

    with pytest.raises(RuntimeError, match="stop after lock check"):
        module.run_pipeline(
            CONFIG_PATH,
            quant_data_dir=tmp_path / "quant_data",
            report_dir=report_dir,
            as_of_date="2026-07-15",
            run_id="lock-lifetime",
            stage_runner=runner,
            git_state_provider=clean_git_state,
        )

    assert observed_locked == [True]
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def test_final_identity_guard_does_not_swallow_body_integrity_error(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    hashes = write_pipeline_snapshots(module, quant_data)
    module.runtime_provider_identity = lambda provider, key: {
        "provider_key": key,
        "provider_tree_sha256": "d" * 64,
    }
    state = {
        "git_commit": "a" * 40,
        "config_hash": module.file_sha256(CONFIG_PATH),
        "data_manifest_hash": None,
        "provider_uri": str(tmp_path / "runtime-provider"),
        "provider_key": "provider-key",
        "provider_tree_sha256": "d" * 64,
        **hashes,
    }

    with pytest.raises(module.MembershipIntegrityError, match="body failed"):
        with module._final_identity_guard(
            state=state,
            config_path=CONFIG_PATH,
            quant_data_dir=quant_data,
            pit_dir=quant_data / "reference" / "pit_csi800",
            reference_dir=quant_data / "reference",
            membership_start="2014-01-02",
            as_of_date="2026-07-15",
            git_state_provider=clean_git_state,
        ) as failure:
            assert failure is None
            raise module.MembershipIntegrityError("body_error", "body failed")


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("config", "config_changed"),
        ("manifest", "data_manifest_changed"),
        ("membership", "membership_snapshot_changed"),
        ("reference", "reference_snapshot_changed"),
        ("provider", "provider_snapshot_changed"),
    ],
)
def test_final_identity_guard_rejects_each_changed_input(
    tmp_path: Path,
    mutation: str,
    expected_reason: str,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    manifest = quant_data / "DATA_MANIFEST.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"version": 1}\n', encoding="utf-8")
    hashes = write_pipeline_snapshots(module, quant_data)
    config_path = tmp_path / "robust_strategy.yaml"
    config_path.write_bytes(CONFIG_PATH.read_bytes())
    state = {
        "git_commit": "a" * 40,
        "config_hash": module.file_sha256(config_path),
        "data_manifest_hash": module.file_sha256(manifest),
        "provider_uri": str(tmp_path / "runtime-provider"),
        "provider_key": "provider-key",
        "provider_tree_sha256": "d" * 64,
        **hashes,
    }
    module.runtime_provider_identity = lambda provider, key: {
        "provider_key": key,
        "provider_tree_sha256": "e" * 64 if mutation == "provider" else "d" * 64,
    }
    if mutation == "config":
        config_path.write_text(config_path.read_text(encoding="utf-8") + "\n# changed\n")
    elif mutation == "manifest":
        manifest.write_text('{"version": 2}\n', encoding="utf-8")
    elif mutation == "membership":
        path = quant_data / "reference" / "pit_csi800" / "daily" / "2026.parquet"
        path.write_bytes(path.read_bytes() + b"membership changed")
    elif mutation == "reference":
        path = quant_data / "reference" / "security_master.parquet"
        path.write_bytes(path.read_bytes() + b"reference changed")

    with module._final_identity_guard(
        state=state,
        config_path=config_path,
        quant_data_dir=quant_data,
        pit_dir=quant_data / "reference" / "pit_csi800",
        reference_dir=quant_data / "reference",
        membership_start="2014-01-02",
        as_of_date="2026-07-15",
        git_state_provider=clean_git_state,
    ) as reason:
        assert reason == expected_reason
