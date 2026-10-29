import json
from quant_workflow.tests.test_qmt_native_training import frozen_input


def test_check_reads_the_selected_frozen_pool_and_reports_gaps(frozen_input, capsys):
    from quant_workflow.scripts.check_qlib_data import main
    directory, frame, calendar, symbols = frozen_input
    assert main(["--input-dir", str(directory)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["provider"] == "qmt_native"
    assert result["rows"] == len(frame)
    assert result["universe"] == symbols
    assert result["calendar"]["count"] == len(calendar)
    assert result["feature_rows"] > 0
    assert "input_dir" not in result
    assert {path.name for path in directory.iterdir()} == {"bars.parquet", "dataset.json"}


def test_direct_provider_entry_reuses_frozen_input_check(frozen_input, capsys):
    from quant_workflow.scripts.build_qlib_direct_provider import main
    directory, _, _, _ = frozen_input
    assert main(["--input-dir", str(directory)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "readable"
