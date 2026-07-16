from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_robust_strategy.py"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "robust_strategy.yaml"


def load_module():
    spec = importlib.util.spec_from_file_location("run_robust_strategy", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
        "preflight_robust_data",
        "prepare_robust_provider",
        "run_robust_development",
    ]
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
    assert "--resume" in development_stage.argv
    assert all(
        str(tmp_path / "reports" / "robust_strategy") in " ".join(stage.argv)
        for stage in stages
    )
    joined = " ".join(part for stage in stages for part in stage.argv)
    membership_stage = stages[0]
    assert (
        membership_stage.argv[membership_stage.argv.index("--membership-source") + 1]
        == "auto"
    )
    assert "candidate_pipeline_latest" not in joined
    assert "latest_training_result" not in joined


def test_pipeline_stops_at_membership_permission_block_and_writes_latest_status(
    tmp_path: Path,
) -> None:
    module = load_module()
    calls: list[str] = []
    report_dir = tmp_path / "reports" / "robust_strategy"

    def runner(stage):
        calls.append(stage.name)
        if stage.name == "build_pit_csi800":
            output_dir = Path(stage.argv[stage.argv.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "build_status_latest.json").write_text(
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

    def runner(stage):
        calls.append(stage.name)
        if stage.name == "download_robust_reference":
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "download_robust_reference_latest.json").write_text(
                json.dumps({"status": "completed", "reason_code": "completed"}),
                encoding="utf-8",
            )
            return 0
        if stage.name == "preflight_robust_data":
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "preflight_latest.json").write_text(
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
        quant_data_dir=tmp_path / "quant_data",
        report_dir=report_dir,
        as_of_date="2026-07-15",
        run_id="test-blocked-preflight",
        skip_membership_build=True,
        stage_runner=runner,
    )

    assert calls == ["download_robust_reference", "preflight_robust_data"]
    assert result["status"] == "blocked_data"
    assert result["failed_stage"] == "preflight_robust_data"


def test_pipeline_prepares_isolated_provider_only_after_preflight_passes(
    tmp_path: Path,
) -> None:
    module = load_module()
    calls: list[str] = []
    report_dir = tmp_path / "reports" / "robust_strategy"

    def runner(stage):
        calls.append(stage.name)
        report_dir.mkdir(parents=True, exist_ok=True)
        if stage.name == "download_robust_reference":
            (report_dir / "download_robust_reference_latest.json").write_text(
                json.dumps({"status": "completed", "reason_code": "completed"}),
                encoding="utf-8",
            )
            return 0
        if stage.name == "preflight_robust_data":
            (report_dir / "preflight_latest.json").write_text(
                json.dumps({"status": "ready", "summary": {"failed_checks": []}}),
                encoding="utf-8",
            )
            return 0
        if stage.name == "prepare_robust_provider":
            (report_dir / "provider_latest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "provider_uri": str(tmp_path / "runtime-provider"),
                    }
                ),
                encoding="utf-8",
            )
            return 0
        if stage.name == "run_robust_development":
            (report_dir / "development_latest.json").write_text(
                json.dumps({"status": "research_candidate_ready"}),
                encoding="utf-8",
            )
            return 0
        raise AssertionError(stage.name)

    result = module.run_pipeline(
        CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=report_dir,
        as_of_date="2026-07-15",
        run_id="test-ready",
        skip_membership_build=True,
        stage_runner=runner,
    )

    assert calls == [
        "download_robust_reference",
        "preflight_robust_data",
        "prepare_robust_provider",
        "run_robust_development",
    ]
    assert result["status"] == "research_candidate_ready"


def test_resume_skips_only_stages_recorded_completed(tmp_path: Path) -> None:
    module = load_module()
    report_dir = tmp_path / "reports" / "robust_strategy"
    run_dir = report_dir / "runs" / "resume-me"
    run_dir.mkdir(parents=True)
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    manifest = quant_data / "DATA_MANIFEST.json"
    manifest.write_text('{"version": 1}\n', encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "resume-me",
                "status": "blocked_data",
                "config_hash": module.file_sha256(CONFIG_PATH),
                "git_commit": "a" * 40,
                "data_manifest_hash": module.file_sha256(manifest),
                "as_of_date": "2026-07-15",
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
    calls: list[str] = []

    def runner(stage):
        calls.append(stage.name)
        if stage.name == "preflight_robust_data":
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "preflight_latest.json").write_text(
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

    assert calls == ["preflight_robust_data"]
    assert result["stages"]["build_pit_csi800"]["status"] == "completed"
    assert result["status"] == "blocked_data"


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
                "data_manifest_hash": module.file_sha256(old_manifest),
                "stages": {
                    name: {"status": "completed", "return_code": 0}
                    for name in [
                        "build_pit_csi800",
                        "download_robust_reference",
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
