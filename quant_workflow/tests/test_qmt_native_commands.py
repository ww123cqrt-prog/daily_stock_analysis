"""Retired update commands cannot re-enter the QMT read-only workflow."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


def test_data_cli_exposes_only_read_operations():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'quant_data_cli.py'
    result = subprocess.run([sys.executable, '-B', str(script), '--help'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'status' in result.stdout and 'export' in result.stdout
    assert 'update' not in result.stdout and 'validate-release' not in result.stdout


def test_training_entry_bootstraps_its_provider_outside_checkout(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'train_baseline_lgb.py'
    code = """
import runpy, sys
sys.argv = [sys.argv[1], '--help']
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as error:
    assert error.code == 0
from quant_workflow import quant_data_qlib_provider
print('PROVIDER_IMPORT_OK')
"""
    result = subprocess.run([sys.executable, '-B', '-c', code, str(script)], cwd=tmp_path,
                            env={**os.environ, 'PYTHONPATH': ''}, capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'PROVIDER_IMPORT_OK' in result.stdout


def test_daily_check_can_use_frozen_input_without_qmt(tmp_path, monkeypatch, capsys):
    from quant_workflow.scripts import run_daily_update as daily
    frame = pd.DataFrame({'date': pd.to_datetime(['2026-10-23'])})
    frame.attrs = {'experiment_id': 'frozen-test', 'actual_end': '2026-10-23'}
    monkeypatch.setattr(daily, 'load_dataset', lambda path: frame)
    monkeypatch.setattr(daily.QMTClient, 'from_env', lambda: (_ for _ in ()).throw(
        AssertionError('offline check must not connect to QMT')))
    assert daily.main(['--input-dir', str(tmp_path), '--required-through', '2026-10-23']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'ready' and result['read_only']
    assert result['experiment_id'] == 'frozen-test'


def test_daily_check_reports_insufficient_dates(tmp_path, monkeypatch, capsys):
    from quant_workflow.scripts import run_daily_update as daily
    frame = pd.DataFrame()
    frame.attrs = {'experiment_id': 'frozen-test', 'actual_end': '2026-10-22'}
    monkeypatch.setattr(daily, 'load_dataset', lambda path: frame)
    assert daily.main(['--input-dir', str(tmp_path), '--required-through', '2026-10-23']) == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'insufficient_dates'
