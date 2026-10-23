"""Exercise Qlib's real writer and the training wrapper's MLflow reader."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pandas as pd
import pytest
import yaml


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("qlib") is None
    or importlib.util.find_spec("mlflow") is None,
    reason="requires the Qlib/MLflow local workflow runtime",
)


@pytest.mark.parametrize(
    ("store_name", "config_kind"),
    [
        ("mlruns", "raw"),
        ("mlruns", "generated_race"),
        ("mlruns #hash", "generated"),
        ("mlruns %20", "generated"),
    ],
)
def test_run_qrun_workflow_and_reader_share_requested_store(tmp_path, store_name, config_kind):
    initial_entries = {item.name for item in tmp_path.iterdir()}
    initial_files = {item.name: item.read_bytes() for item in tmp_path.iterdir() if item.is_file()}
    # StaticDataLoader keeps the real model/dataset/SignalRecord workflow entirely
    # local while a subprocess isolates Qlib's global experiment manager.
    provider = tmp_path / "provider"
    provider.mkdir()
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=4), ["SH600000", "SH600001", "SH600002"]],
        names=["datetime", "instrument"],
    )
    data = pd.DataFrame(
        [[float(value), float(value * 2)] for value in range(1, 13)],
        index=index,
        columns=pd.MultiIndex.from_tuples([("feature", "x"), ("label", "y")]),
    )
    data_path = tmp_path / "data.pkl"
    data.to_pickle(data_path)
    config = {
        "qlib_init": {"provider_uri": str(provider), "region": "cn", "kernels": 1},
        "task": {
            "model": {"class": "LinearModel", "module_path": "qlib.contrib.model.linear"},
            "dataset": {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "DataHandlerLP",
                        "module_path": "qlib.data.dataset.handler",
                        "kwargs": {
                            "data_loader": {
                                "class": "StaticDataLoader",
                                "module_path": "qlib.data.dataset.loader",
                                "kwargs": {"config": str(data_path)},
                            }
                        },
                    },
                    "segments": {
                        "train": ["2024-01-02", "2024-01-03"],
                        "test": ["2024-01-04", "2024-01-05"],
                    },
                },
            },
            "record": [{"class": "SignalRecord", "module_path": "qlib.workflow.record_temp"}],
        },
    }
    config_path = tmp_path / "workflow.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    store = tmp_path / store_name
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_baseline_lgb.py"
    code = textwrap.dedent(
        """
        import importlib.util
        import json
        import os
        from pathlib import Path
        import sys
        from unittest.mock import patch

        import mlflow
        import pandas as pd

        lock_paths = []
        def audit_filelocks(event, arguments):
            if event == "open" and isinstance(arguments[0], (str, bytes)):
                path = Path(os.fsdecode(arguments[0]))
                if path.name == "filelock":
                    lock_paths.append(str(path.resolve()))
        sys.addaudithook(audit_filelocks)

        script, config_path, store_path = map(Path, sys.argv[1:4])
        spec = importlib.util.spec_from_file_location("train_baseline_lgb", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if sys.argv[4] != "raw":
            # The static fixture needs no governed market-data Provider. Keep
            # the experiment manager and the complete Qlib workflow real.
            with patch.object(module, "qlib_workflow_init_config", return_value={
                "provider_uri": str(config_path.parent / "provider"),
                "region": "cn", "kernels": 1,
            }):
                config_path = module.write_runtime_config_with_provider(
                    config_path, config_path.parent / "provider", "canonical",
                    config_path.parent / "runtime", extend_dates_to_provider_latest=False,
                )
        if sys.argv[4] == "generated_race":
            # Another creator can win between the wrapper's lookup and create.
            create_experiment = mlflow.tracking.MlflowClient.create_experiment
            def competing_create(client, *arguments, **keywords):
                create_experiment(client, *arguments, **keywords)
                return create_experiment(client, *arguments, **keywords)
            with patch.object(mlflow.tracking.MlflowClient, "create_experiment", competing_create):
                run_id = module.run_qrun(str(config_path), experiment_name="tracking-uri-workflow")
        else:
            run_id = module.run_qrun(str(config_path), experiment_name="tracking-uri-workflow")
        client = mlflow.tracking.MlflowClient(tracking_uri=store_path.resolve().as_uri())
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED"
        artifacts = {item.path for item in client.list_artifacts(run_id)}
        assert {"task", "params.pkl", "dataset", "pred.pkl", "label.pkl"} <= artifacts
        artifact_root = store_path / run.info.experiment_id / run_id / "artifacts"
        prediction = pd.read_pickle(artifact_root / "pred.pkl")
        labels = pd.read_pickle(artifact_root / "label.pkl")
        expected = [14.0, 16.0, 18.0, 20.0, 22.0, 24.0]
        assert all(abs(actual - want) < 1e-10 for actual, want in zip(prediction["score"], expected))
        assert len(prediction) == 6
        assert labels.iloc[:, 0].tolist() == expected
        unexpected_locks = [name for name in lock_paths if not Path(name).is_relative_to(store_path.resolve())]
        assert not unexpected_locks, unexpected_locks
        print(json.dumps({"run_id": run_id, "store": str(store_path), "status": run.info.status,
                          "lock_paths": lock_paths}))
        """
    )
    env = dict(
        os.environ,
        QUANT_WORKFLOW_MLRUNS_DIR=str(store),
        MLFLOW_TRACKING_URI=store.resolve().as_uri(),
        MLFLOW_ALLOW_FILE_STORE="true",
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(script), str(config_path), str(store), config_kind],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout.splitlines()[-1])
    assert summary["status"] == "FINISHED"
    assert summary["store"] == str(store)
    assert sorted(item.name for item in tmp_path.iterdir() if item.name.startswith("mlruns")) == [
        store_name
    ]
    expected_entries = {"provider", "data.pkl", "workflow.yaml", store_name}
    if config_kind != "raw":
        expected_entries.add("runtime")
    assert {item.name for item in tmp_path.iterdir()} == initial_entries | expected_entries
    for name, contents in initial_files.items():
        assert (tmp_path / name).read_bytes() == contents


@pytest.mark.parametrize("error_code", ["PERMISSION_DENIED", "RESOURCE_ALREADY_EXISTS"])
def test_run_qrun_preserves_experiment_creation_error(tmp_path, monkeypatch, error_code):
    import mlflow
    from mlflow.protos import databricks_pb2

    monkeypatch.setenv("QUANT_WORKFLOW_MLRUNS_DIR", str(tmp_path / "mlruns"))
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_baseline_lgb.py"
    spec = importlib.util.spec_from_file_location("train_baseline_lgb", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = mlflow.exceptions.MlflowException(
        "create failed", error_code=getattr(databricks_pb2, error_code)
    )
    assert expected.error_code == error_code

    def fail_creation(*_arguments, **_keywords):
        raise expected

    monkeypatch.setattr(mlflow.tracking.MlflowClient, "create_experiment", fail_creation)
    with pytest.raises(mlflow.exceptions.MlflowException) as caught:
        module.run_qrun(str(tmp_path / "unused.yaml"), experiment_name="failed-creation")
    assert caught.value is expected
