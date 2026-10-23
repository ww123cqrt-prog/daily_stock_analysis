"""Small synthetic suites verify the thin pytest entry; no business data."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from datetime import datetime

from scripts.testing import block_runner as runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        # Synthetic repositories must not inherit the outer runner's index.
        self.enterContext(patch.dict(os.environ))
        os.environ.pop('GIT_INDEX_FILE', None)

    def _source_manifest(self, parent, suite):
        files = {}
        for path in suite.rglob('*'):
            if path.is_file() and (path.suffix == '.py' or path.name == 'setup.cfg'):
                relative = path.relative_to(suite).as_posix()
                content = path.read_bytes()
                files[relative] = dict(sha256=hashlib.sha256(content).hexdigest(), size=len(content))
        manifest = parent / 'source-manifest.json'
        manifest.write_text(json.dumps(dict(base_commit='0' * 40, files=files)))
        return manifest

    def _manifest_main(self, suite, destination, manifest, workers=0, args=()):
        env = dict(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_PLUGINS='xdist.plugin')
        with patch.dict(os.environ, env), patch.object(runner, '__file__', str(suite / 'scripts/testing/block_runner.py')):
            return runner.main(['--run-dir', str(destination), '--source-manifest', str(manifest),
                                '--workers', str(workers), '--wall-timeout', '10', '--', *args])

    def test_manifest_hashes_source_and_ignores_generated_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            suite = self._wall_suite(parent, 'def test_pass():\n    pass\n')
            manifest = self._source_manifest(parent, suite)
            before = runner.source_version(suite, manifest)
            self.assertTrue(before['complete'], before)
            for relative in ('output/test_export.py', 'quant_workflow/reports/model.py', 'quant_workflow/mlruns/model.py',
                             'quant_workflow/scripts/backtest_results/test_model.py'):
                path = suite / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('raise RuntimeError("generated output is not a test input")\n')
            self.assertEqual(before, runner.source_version(suite, manifest))
            (suite / 'tests/test_synthetic.py').write_text('def test_pass():\n    assert False\n')
            changed = runner.source_version(suite, manifest)
            self.assertFalse(changed['complete'])
            self.assertNotEqual(before, changed)

    def test_manifest_refuses_unlisted_runtime_source_and_tests(self):
        for relative in ('tests/test_extra.py', 'quant_workflow/tests/test_extra.py', 'src/extra.py', 'scripts/draft.py',
                         'sitecustomize.py', 'main.py'):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temp:
                parent = Path(temp)
                suite = self._wall_suite(parent, 'def test_pass():\n    pass\n')
                manifest = self._source_manifest(parent, suite)
                path = suite / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('def test_extra():\n    pass\n')
                version = runner.source_version(suite, manifest)
                self.assertFalse(version['complete'], version)
                self.assertIn(relative, '\n'.join(version['problems']))

    def test_manifest_refuses_unsafe_paths_and_declared_outputs(self):
        for relative in ('../outside.py', '/outside.py', 'data/formal.py', 'quant_workflow/reports/model.py'):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temp:
                parent = Path(temp)
                suite = self._wall_suite(parent, 'def test_pass():\n    pass\n')
                manifest = self._source_manifest(parent, suite)
                document = json.loads(manifest.read_text())
                document['files'][relative] = dict(sha256='0' * 64, size=0)
                manifest.write_text(json.dumps(document))
                self.assertFalse(runner.source_version(suite, manifest)['complete'])

    def test_manifest_verifies_relative_symlink_target_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            suite = self._wall_suite(parent, 'def test_pass():\n    pass\n')
            manifest = self._source_manifest(parent, suite)
            (suite / 'AGENTS.md').write_text('local rules\n')
            link = suite / 'CLAUDE.md'
            try:
                link.symlink_to('AGENTS.md')
            except OSError as exc:
                self.skipTest(f'native symlink creation unavailable: {exc}')
            document = json.loads(manifest.read_text())
            document['files']['CLAUDE.md'] = dict(mode='120000', size=9,
                                                sha256=hashlib.sha256(b'AGENTS.md').hexdigest())
            manifest.write_text(json.dumps(document))
            version = runner.source_version(suite, manifest)
            self.assertTrue(version['complete'], version)
            self.assertEqual(version['files']['CLAUDE.md']['target'], 'AGENTS.md')
            link.unlink()
            link.symlink_to('tests/test_synthetic.py')
            self.assertFalse(runner.source_version(suite, manifest)['complete'])
            link.unlink()
            (parent / 'outside.py').write_text('pass\n')
            link.symlink_to('../outside.py')
            self.assertFalse(runner.source_version(suite, manifest)['complete'])

    def test_manifest_parallel_round_and_serial_failure_confirmation(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            suite = self._wall_suite(parent, '''from pathlib import Path
def test_once():
    marker = Path(__file__).with_suffix('.marker')
    existed = marker.exists()
    marker.touch()
    assert existed
def test_other():
    marker = Path(__file__).with_suffix('.other')
    assert not marker.exists()
    marker.touch()
''')
            manifest = self._source_manifest(parent, suite)
            run = parent / 'run'
            self.assertEqual(self._manifest_main(suite, run, manifest, workers=2), 1)
            outcome = json.loads((run / 'outcome.json').read_text())
            self.assertTrue(outcome['complete'], outcome)
            self.assertEqual(outcome['not_reproduced'], ['tests/test_synthetic.py::test_once'])
            first = json.loads((run / 'first/start.json').read_text())['command']
            second = json.loads((run / 'rerun/start.json').read_text())['command']
            self.assertEqual(first[first.index('-n') + 1], '2')
            self.assertEqual(second[second.index('-n') + 1], '0')

    def test_manifest_source_change_blocks_serial_confirmation(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            suite = self._wall_suite(parent, '''from pathlib import Path
def test_changes_source():
    path = Path(__file__)
    path.write_text(path.read_text() + '\\n')
    assert False
''')
            manifest = self._source_manifest(parent, suite)
            run = parent / 'run'
            self.assertEqual(self._manifest_main(suite, run, manifest), 2)
            self.assertFalse((run / 'rerun').exists())
            outcome = json.loads((run / 'outcome.json').read_text())
            self.assertIn('source version changed before rerun', outcome['problems'])

    def test_manifest_rejects_explicit_unlisted_test_before_body(self):
        for workers in (0, 2):
            self._manifest_rejects_explicit_test(workers)

    def _manifest_rejects_explicit_test(self, workers):
        with self.subTest(workers=workers), tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            suite = self._wall_suite(parent, 'def test_pass():\n    pass\n')
            manifest = self._source_manifest(parent, suite)
            external = suite / 'extra/test_external.py'
            external.parent.mkdir()
            external.write_text('from pathlib import Path\ndef test_external():\n    Path(__file__).with_suffix(".ran").touch()\n')
            run = parent / 'run'
            self.assertEqual(self._manifest_main(suite, run, manifest, workers=workers, args=['extra/test_external.py']), 2)
            self.assertFalse(external.with_suffix('.ran').exists())
            if not workers:
                self.assertIn('source manifest', (run / 'first/stderr.log').read_text())

    def _wall_suite(self, parent, source):
        suite = parent / 'checkout'
        suite.mkdir()
        (suite / 'setup.cfg').write_text('[tool:pytest]\ntestpaths=tests\n')
        shutil.copytree(Path(__file__).resolve().parents[1] / 'scripts/testing', suite / 'scripts/testing')
        shutil.copyfile(runner.__file__, suite / 'scripts/testing/block_runner.py')
        (suite / 'scripts/__init__.py').touch()
        (suite / 'tests').mkdir()
        (suite / 'tests/test_synthetic.py').write_text(source)
        return suite

    def _wall_main(self, suite, destination, seconds, workers=0):
        version = dict(head='synthetic', diff='', status='', untracked=[])
        env = dict(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_PLUGINS='xdist.plugin')
        with patch.dict(os.environ, env), patch.object(runner, '__file__', str(suite / 'scripts/testing/block_runner.py')), patch.object(
            runner, 'source_version', return_value=version
        ):
            code = runner.main(['--run-dir', str(destination), '--workers', str(workers), '--wall-timeout', str(seconds)])
            for log in destination.glob('*/*.log'):
                if log.stat().st_size:
                    print(str(log), log.read_text(errors='replace'))
            return code

    def test_wall_timeout_terminal_is_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            suite = self._wall_suite(parent, '''from pathlib import Path
import time
def test_wait():
    Path(__file__).with_suffix('.ready').touch()
    time.sleep(30)
''')
            destination = parent / 'run'
            self.assertEqual(self._wall_main(suite, destination, 1.5), 2)
            self.assertTrue((suite / 'tests/test_synthetic.ready').exists(), 'timeout must reach the test body')
            first = json.loads((destination / 'first/summary.json').read_text())
            terminal = json.loads((destination / 'first/exit.json').read_text())
            outcome = json.loads((destination / 'outcome.json').read_text())
            self.assertTrue(first['timed_out'], first)
            self.assertEqual(first['termination_reason'], 'wall_timeout')
            self.assertFalse(first['complete'])
            self.assertFalse(outcome['complete'])
            self.assertFalse((destination / 'rerun').exists())
            self.assertFalse((destination / 'first/junit.xml').exists())
            self.assertLess(terminal['wall_seconds'], 15)
            self.assertTrue(terminal['cleanup']['complete'], terminal)
            self.assertNotEqual(terminal['exitcode'], 124)
            self.assertEqual(first['exitcode'], terminal['exitcode'])
            start = json.loads((destination / 'first/start.json').read_text())
            self.assertEqual(start['wall_timeout_seconds'], 1.5)
            self.assertEqual(start['timeout_scope'], 'one_pytest_invocation')

    def test_normal_completion_keeps_real_n0_evidence(self):
        import pytest
        import xdist

        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            suite = self._wall_suite(parent, '''import json,os,sys
from pathlib import Path
import pytest,xdist
def test_pass():
    Path(__file__).with_suffix('.runtime').write_text(json.dumps(dict(
        pid=os.getpid(), executable=sys.executable, prefix=sys.prefix,
        pytest=pytest.__file__, xdist=xdist.__file__)))
    assert 1 + 1 == 2
''')
            destination = parent / 'run'
            self.assertEqual(self._wall_main(suite, destination, 5), 0)
            terminal = json.loads((destination / 'first/exit.json').read_text())
            first = json.loads((destination / 'first/summary.json').read_text())
            self.assertEqual(terminal['exitcode'], 0)
            self.assertEqual(terminal['termination_reason'], 'completed')
            self.assertFalse(terminal['timed_out'])
            self.assertEqual(terminal['cleanup']['signals'], [])
            self.assertTrue(terminal['cleanup']['complete'])
            self.assertTrue(first['complete'], first)
            self.assertEqual(set(first['statuses'].values()), {'passed'})
            self.assertTrue((destination / 'first/junit.xml').is_file())
            events = [json.loads(line) for line in (destination / 'first/checkpoint.jsonl').read_text().splitlines()]
            self.assertEqual([event['exitstatus'] for event in events if event['event'] == 'session_finish'], [0])
            start = json.loads((destination / 'first/start.json').read_text())
            self.assertEqual([event['pid'] for event in events if event['event'] == 'session_start'], [start['pid']])
            runtime = json.loads((suite / 'tests/test_synthetic.runtime').read_text())
            self.assertEqual(runtime['pid'], start['pid'])
            self.assertEqual(Path(runtime['executable']), Path(sys.executable))
            self.assertEqual(Path(runtime['prefix']), Path(sys.prefix))
            for module in (pytest, xdist):
                self.assertEqual(Path(runtime[module.__name__]).resolve(), Path(module.__file__).resolve())

    @unittest.skipUnless(os.name == 'posix', 'POSIX session group capability')
    def test_exited_leader_child_pipe_and_term_ignore_are_cleaned(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            suite = self._wall_suite(parent, '''import os
from pathlib import Path
import subprocess
import sys
import time
def test_leave_child():
    marker = Path(__file__).with_suffix('.child')
    child_code = "import os, signal, time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); Path(" + repr(str(marker)) + ").write_text(str(os.getpid())); time.sleep(20)"
    child = subprocess.Popen([sys.executable, '-B', '-c', child_code], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    deadline = time.monotonic() + 3
    while not marker.exists():
        assert time.monotonic() < deadline
        time.sleep(.01)
    os._exit(23)
''')
            control = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(20)'],
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                destination = parent / 'run'
                self.assertEqual(self._wall_main(suite, destination, 10), 2)
                terminal = json.loads((destination / 'first/exit.json').read_text())
                self.assertEqual(terminal['exitcode'], 23)
                self.assertFalse(terminal['timed_out'])
                self.assertEqual(terminal['termination_reason'], 'residual_descendants')
                self.assertTrue(terminal['cleanup']['residual_after_leader_exit'])
                self.assertEqual(terminal['cleanup']['signals'], ['SIGTERM', 'SIGKILL'])
                self.assertTrue(terminal['cleanup']['complete'], terminal)
                self.assertEqual(terminal['cleanup']['remaining'], [])
                print(json.dumps(dict(synthetic_case='leader_child_pipe', terminal=terminal)))
                self.assertIsNone(control.poll(), 'independent owned sibling must not be signaled')
                child_pid = int((suite / 'tests/test_synthetic.child').read_text())
                child = subprocess.run(['/bin/ps', '-p', str(child_pid), '-o', 'pid='],
                                       capture_output=True, text=True, timeout=2)
                self.assertFalse(child.stdout.strip(), child.stdout)
            finally:
                if control.poll() is None:
                    control.terminate()
                try:
                    control.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    control.kill()
                    control.wait(timeout=2)

    @unittest.skipUnless(os.name == 'posix', 'POSIX TERM and session group capability')
    def test_runner_term_cleans_separate_pytest_session(self):
        self._runner_term_case(group_term=False)

    @unittest.skipUnless(os.name == 'posix', 'POSIX TERM group and session group capability')
    def test_runner_group_term_cleans_separate_pytest_session(self):
        self._runner_term_case(group_term=True)

    def _runner_term_case(self, group_term):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            suite = self._wall_suite(parent, '''from pathlib import Path
import time
def test_wait():
    Path(__file__).with_suffix('.ready').touch()
    time.sleep(30)
''')
            destination = parent / 'run'
            env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_PLUGINS='xdist.plugin')
            env['PYTHONPATH'] = os.pathsep.join([str(suite), env.get('PYTHONPATH', '')])
            launcher = "from scripts.testing import block_runner as r; r.source_version=lambda root: dict(head='synthetic', diff='', status='', untracked=[]); raise SystemExit(r.main())"
            with (parent / 'owner.stdout').open('wb') as stdout, (parent / 'owner.stderr').open('wb') as stderr:
                owner = subprocess.Popen([sys.executable, '-B', '-c', launcher, '--run-dir', str(destination),
                                          '--workers', '0', '--wall-timeout', '8'], cwd=suite, env=env,
                                         stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
                try:
                    deadline = time.monotonic() + 5
                    while not (suite / 'tests/test_synthetic.ready').exists():
                        self.assertIsNone(owner.poll(), (parent / 'owner.stderr').read_text())
                        self.assertLess(time.monotonic(), deadline, 'bounded test readiness')
                        time.sleep(.02)
                    if group_term:
                        self.assertEqual(os.getpgid(owner.pid), owner.pid)
                        self.assertEqual(os.getsid(owner.pid), owner.pid)
                        os.killpg(owner.pid, runner.signal.SIGTERM)
                    else:
                        owner.terminate()  # Only the exact Popen child; never SIGINT/CRT.
                    self.assertEqual(owner.wait(timeout=18), 2)
                finally:
                    if owner.poll() is None:
                        owner.terminate()
                        owner.wait(timeout=18)
            terminal = json.loads((destination / 'first/exit.json').read_text())
            first = json.loads((destination / 'first/summary.json').read_text())
            self.assertEqual(terminal['termination_reason'], 'interrupted')
            self.assertFalse(terminal['timed_out'])
            self.assertTrue(terminal['cleanup']['complete'], terminal)
            print(json.dumps(dict(synthetic_case='runner_term', terminal=terminal)))
            self.assertFalse(first['complete'])
            self.assertFalse((destination / 'rerun').exists())
            start = json.loads((destination / 'first/start.json').read_text())
            self.assertNotEqual(start['pgid'], owner.pid)
            remaining = subprocess.run(['/bin/ps', '-axo', 'pgid='], capture_output=True, text=True, timeout=2)
            self.assertNotIn(str(start['pgid']), remaining.stdout.split())

    @unittest.skipUnless(os.name == 'posix', 'POSIX ownership check')
    def test_unknown_group_and_permission_error_stop_signals(self):
        child = Mock(pid=999999, returncode=None)
        with patch.object(runner, 'group_members', side_effect=RuntimeError('identity changed')), patch.object(runner.os, 'killpg') as kill:
            result = runner.cleanup_owned(child, {child.pid: 'birth'})
            self.assertFalse(result['complete'])
            kill.assert_not_called()
        with patch.object(runner, 'group_members', return_value={child.pid: ('birth', 'S')}), patch.object(
            runner.os, 'killpg', side_effect=PermissionError('synthetic refusal')
        ) as kill:
            result = runner.cleanup_owned(child, {child.pid: 'birth'})
            self.assertFalse(result['complete'])
            self.assertEqual(kill.call_count, 1)
            self.assertIn('synthetic refusal', result['problems'])
        with patch.object(runner.os, 'killpg') as kill:
            result = runner.cleanup_owned(child, {})
            self.assertFalse(result['complete'])
            kill.assert_not_called()

    def test_direct_child_capability_is_bounded(self):
        child = Mock(pid=999999, returncode=None)
        child.poll.side_effect = [None, None]
        def finish(*, timeout):
            self.assertEqual(timeout, 5)
            if child.kill.called:
                child.returncode = -9
            else:
                raise subprocess.TimeoutExpired('synthetic', timeout)
        child.wait.side_effect = finish
        with patch.object(runner.os, 'name', 'nt'):
            result = runner.cleanup_owned(child, {})
        self.assertEqual(result['scope'], 'direct_child')
        self.assertTrue(result['complete'])
        self.assertEqual(result['signals'], ['terminate', 'kill'])
        self.assertEqual(child.wait.call_count, 2)

    def test_windows_job_setup_failure_does_not_start_pytest(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            suite = self._wall_suite(parent, 'from pathlib import Path\ndef test_started():\n    Path(__file__).with_suffix(".started").touch()\n')
            destination = parent / 'run'
            env = dict(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_PLUGINS='xdist.plugin')
            with patch.dict(os.environ, env), patch.object(runner.os, 'name', 'nt'), patch.object(
                runner, 'windows_job', side_effect=PermissionError('synthetic job refusal'), create=True
            ):
                result = runner.execute(suite, destination, parent / 'cache', [], 0, 'worksteal', wall_timeout=5)
            self.assertEqual(result['termination_reason'], 'execution_error', result)
            self.assertFalse(result['complete'])
            self.assertFalse((suite / 'tests/test_synthetic.started').exists())
            self.assertTrue(result['cleanup']['complete'], result)
            self.assertIn('synthetic job refusal', result['problems'])
            # Replaying the real bootstrap with EOF must still refuse pytest.
            launch = json.loads((destination / 'start.json').read_text())['launch_command']
            refusal = subprocess.run(launch, cwd=suite, stdin=subprocess.DEVNULL,
                                     capture_output=True, timeout=5)
            self.assertEqual(refusal.returncode, 2, refusal.stderr)
            self.assertFalse((suite / 'tests/test_synthetic.started').exists())

    def test_windows_venv_job_owns_interpreter_and_keeps_venv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            child = Mock(pid=999999, returncode=0)
            kernel = Mock()
            with patch.object(runner.os, 'name', 'nt'), patch.object(runner.sys, 'executable', '/venv/Scripts/python.exe'), patch.object(
                runner.sys, '_base_executable', '/base/python.exe'
            ), patch.object(runner.subprocess, 'Popen', return_value=child) as launch, patch.object(
                runner, 'windows_job', return_value=(kernel, 123)
            ), patch.object(runner, 'windows_active', return_value=0), patch.object(runner, 'wait_owned', return_value='completed'):
                result = runner.execute(root, root / 'round', root / 'cache', [], 0, 'worksteal')
            self.assertEqual(launch.call_args.args[0][0], '/base/python.exe')
            self.assertEqual(launch.call_args.kwargs['env']['__PYVENV_LAUNCHER__'], '/venv/Scripts/python.exe')
            self.assertTrue(result['cleanup']['complete'], result)

    def test_windows_job_cleanup_checks_descendants_after_leader_exit(self):
        child = Mock(pid=999999, returncode=23)
        kernel = Mock()
        with patch.object(runner.os, 'name', 'nt'), patch.object(
            runner, 'windows_active', side_effect=[2, 0], create=True
        ):
            result = runner.cleanup_owned(child, {}, job=(kernel, 123))
        self.assertEqual(result['scope'], 'windows_job')
        self.assertTrue(result['complete'], result)
        self.assertTrue(result['residual_after_leader_exit'])
        self.assertEqual(result['signals'], ['TerminateJobObject'])
        self.assertEqual(result['active_processes'], 0)
        self.assertEqual(result['remaining'], [])

    def test_windows_job_query_failure_never_claims_cleanup_complete(self):
        child = Mock(pid=999999, returncode=0)
        kernel = Mock()
        with patch.object(runner.os, 'name', 'nt'), patch.object(
            runner, 'windows_active', side_effect=OSError('synthetic job query refusal'), create=True
        ):
            result = runner.cleanup_owned(child, {}, job=(kernel, 123))
        self.assertFalse(result['complete'], result)
        self.assertIn('synthetic job query refusal', result['problems'])
        self.assertIsNone(result['remaining'])
        kernel.CloseHandle.assert_called_once_with(123)

    def test_windows_closed_gate_still_cleans_the_owned_job(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            child = Mock(pid=999999, returncode=2)
            child.stdin.close.side_effect = BrokenPipeError('synthetic closed gate')
            kernel = Mock()
            with patch.object(runner.os, 'name', 'nt'), patch.object(runner.subprocess, 'Popen', return_value=child), patch.object(
                runner, 'windows_job', return_value=(kernel, 123)
            ), patch.object(runner, 'windows_active', return_value=0):
                result = runner.execute(root, root / 'round', root / 'cache', [], 0, 'worksteal')
            self.assertEqual(result['termination_reason'], 'execution_error')
            self.assertFalse(result['complete'])
            self.assertEqual(result['cleanup']['scope'], 'windows_job')
            self.assertTrue(result['cleanup']['complete'], result)
            kernel.CloseHandle.assert_called_once_with(123)

    @unittest.skipUnless(os.name == 'nt', 'native Windows Job Object capability')
    def test_windows_xdist_timeout_reaps_workers_and_their_children(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            suite = self._wall_suite(parent, '''import os
from pathlib import Path
import subprocess
import sys
import time
def wait_with_child():
    marker = Path(__file__).with_name(os.environ['PYTEST_XDIST_WORKER'] + '.child')
    child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(30)'], stdin=subprocess.DEVNULL)
    marker.write_text(str(child.pid))
    time.sleep(30)
def test_a(): wait_with_child()
def test_b(): wait_with_child()
def test_c(): wait_with_child()
def test_d(): wait_with_child()
''')
            control = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(30)'], stdin=subprocess.DEVNULL)
            try:
                destination = parent / 'run'
                self.assertEqual(self._wall_main(suite, destination, 5, workers=2), 2)
                terminal = json.loads((destination / 'first/exit.json').read_text())
                self.assertEqual(len(list((suite / 'tests').glob('gw*.child'))), 2)
                self.assertEqual(terminal['termination_reason'], 'wall_timeout')
                self.assertEqual(terminal['cleanup']['scope'], 'windows_job')
                self.assertTrue(terminal['cleanup']['complete'], terminal)
                self.assertEqual(terminal['cleanup']['active_processes'], 0)
                self.assertIsNone(control.poll(), 'independent sibling must remain alive')
            finally:
                control.terminate()
                control.wait(timeout=5)

    @unittest.skipUnless(os.name == 'nt', 'native Windows Job Object capability')
    def test_windows_exited_leader_still_reaps_its_child(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            suite = self._wall_suite(parent, '''import os
from pathlib import Path
import subprocess
import sys
def test_leave_child():
    child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(30)'], stdin=subprocess.DEVNULL)
    Path(__file__).with_suffix('.child').write_text(str(child.pid))
    os._exit(23)
''')
            destination = parent / 'run'
            self.assertEqual(self._wall_main(suite, destination, 10), 2)
            terminal = json.loads((destination / 'first/exit.json').read_text())
            self.assertTrue((suite / 'tests/test_synthetic.child').is_file())
            self.assertEqual(terminal['exitcode'], 23)
            self.assertEqual(terminal['termination_reason'], 'residual_descendants')
            self.assertTrue(terminal['cleanup']['residual_after_leader_exit'])
            self.assertTrue(terminal['cleanup']['complete'], terminal)
            self.assertEqual(terminal['cleanup']['active_processes'], 0)

    def test_timeout_cleanup_failure_keeps_truthful_terminal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            child = Mock(pid=999999, returncode=None)
            cleanup = dict(scope='posix_session_group', complete=False, signals=[], remaining=[],
                           problems=['synthetic identity unavailable'])
            with patch.object(runner.os, 'name', 'posix'), patch.object(runner.subprocess, 'Popen', return_value=child), patch.object(runner, 'group_members', return_value={child.pid: ('birth', 'S')}), patch.object(
                runner, 'wait_owned', return_value='wall_timeout'
            ), patch.object(runner, 'cleanup_owned', return_value=cleanup):
                result = runner.execute(root, root / 'round', root / 'cache', [], 0, 'worksteal', wall_timeout=.1)
            terminal = json.loads((root / 'round/exit.json').read_text())
            self.assertFalse(result['complete'])
            self.assertTrue(result['timed_out'])
            self.assertEqual(result['termination_reason'], 'cleanup_incomplete')
            self.assertIsNone(result['exitcode'])
            self.assertEqual(terminal['cleanup']['signals'], [])
            self.assertIn('synthetic identity unavailable', result['problems'])
            self.assertFalse((root / 'round/junit.xml').exists())

    @unittest.skipUnless(os.name == 'posix', 'POSIX group identity capability')
    def test_replacement_group_is_not_owned(self):
        child = Mock(pid=999999, returncode=0)
        with patch.object(runner.subprocess, 'run', return_value=Mock(stdout='999999 999999 S later_birth\n')):
            with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                runner.group_members(child, {999999: 'original_birth'})
        with patch.object(runner.subprocess, 'run', return_value=Mock(stdout='')), patch.object(runner.os, 'killpg') as kill:
            result = runner.cleanup_owned(child, {999999: 'original_birth'})
            self.assertTrue(result['complete'])
            kill.assert_not_called()

    @unittest.skipUnless(os.name == 'posix', 'POSIX TERM group status probe')
    def test_term_interrupting_status_probe_preserves_cleanup_path(self):
        child = Mock(pid=999999, returncode=None)
        interruption = [None]
        def stopped_probe(*args, **kwargs):
            interruption[0] = runner.signal.SIGTERM
            return Mock(returncode=-runner.signal.SIGTERM, stdout='')
        with patch.object(runner.subprocess, 'run', side_effect=stopped_probe):
            self.assertEqual(runner.wait_owned(child, time.monotonic() + 1, interruption, {child.pid: 'birth'}), 'interrupted')

    @unittest.skipUnless(os.name == 'posix', 'POSIX TERM status probe error')
    def test_term_flag_does_not_mask_status_probe_error(self):
        child = Mock(pid=999999, returncode=None)
        interruption = [None]
        def failed_probe(*args, **kwargs):
            interruption[0] = runner.signal.SIGTERM
            return Mock(returncode=2, stdout='')
        with patch.object(runner.subprocess, 'run', side_effect=failed_probe), patch.object(runner.os, 'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'cannot observe owned child status'):
                runner.wait_owned(child, time.monotonic() + 1, interruption, {child.pid: 'birth'})
            kill.assert_not_called()

    def test_wall_cap_validation_and_round_forwarding(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            for index, value in enumerate(('0', '-1', 'nan', 'inf', '-inf')):
                destination = parent / str(index)
                with self.assertRaises(SystemExit):
                    runner.main(['--run-dir', str(destination), '--wall-timeout=' + value])
                self.assertFalse(destination.exists())
            suite = parent / 'source'
            suite.mkdir()
            (suite / 'test_one.py').write_text('def test_one():\n    pass\n')
            baseline = dict(head='synthetic', diff='', status='', untracked=[])
            failed = dict(complete=True, failed=['test_one.py::test_one'], collected=['test_one.py::test_one'], statuses={'test_one.py::test_one': 'failed'}, exitcode=1)
            passed = dict(failed, failed=[], statuses={'test_one.py::test_one': 'passed'}, exitcode=0)
            with patch.object(runner, '__file__', str(suite / 'scripts/testing/block_runner.py')), patch.object(runner, 'source_version', return_value=baseline), patch.object(
                runner, 'execute', side_effect=[failed, passed]
            ) as execute:
                self.assertEqual(runner.main(['--run-dir', str(parent / 'rounds'), '--wall-timeout', '.5']), 1)
            self.assertEqual([call.kwargs['wall_timeout'] for call in execute.call_args_list], [.5, .5])

    def test_phase_matrix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            launch = root / 'different-cwd'
            launch.mkdir()
            for scenario in ('pass', 'transient', 'skip-rerun', 'xfail-rerun', 'fail', 'phases', 'collection', 'crash'):
                suite = root / ('acceptance-' + scenario) / 'checkout'
                suite.mkdir(parents=True)
                (suite / 'setup.cfg').write_text('[tool:pytest]\ntestpaths=tests\n')
                shutil.copytree(Path(runner.__file__).parent, suite / 'scripts/testing')
                (suite / 'scripts/__init__.py').touch()
                tests = suite / 'tests'
                tests.mkdir()
                source = {
                    'pass': '''import json
import os
from pathlib import Path
import time

def _record_task(task):
    root = Path(__file__).parent
    worker = os.environ['PYTEST_XDIST_WORKER']
    started = time.monotonic()
    (root / (worker + '.ready')).touch()
    deadline = started + 10
    while len(list(root.glob('gw*.ready'))) != 2:
        assert time.monotonic() < deadline, 'both workers must execute a task'
        time.sleep(0.01)
    time.sleep(0.05)
    (root / (task + '.json')).write_text(json.dumps({
        'task': task, 'worker': worker, 'pid': os.getpid(),
        'started': started, 'ended': time.monotonic(),
    }))

def test_task_a():
    _record_task('a')
def test_task_b():
    _record_task('b')
def test_task_c():
    _record_task('c')
def test_task_d():
    _record_task('d')
''',
                    'transient': "from pathlib import Path\ndef test_once():\n    marker=Path(__file__).with_suffix('.marker')\n    existed=marker.exists()\n    marker.touch()\n    assert existed\n",
                    'skip-rerun': "from pathlib import Path\nimport pytest\ndef test_once():\n    marker=Path(__file__).with_suffix('.marker')\n    if marker.exists():\n        pytest.skip('skip on rerun')\n    marker.touch()\n    assert False\n",
                    'xfail-rerun': "from pathlib import Path\nimport pytest\ndef test_once():\n    marker=Path(__file__).with_suffix('.marker')\n    if marker.exists():\n        pytest.xfail('xfail on rerun')\n    marker.touch()\n    assert False\n",
                    'fail': 'def test_fail():\n    assert False\n',
                    'phases': '''import pytest
@pytest.fixture
def bad_setup():
    raise RuntimeError('setup')
@pytest.fixture
def bad_teardown():
    yield
    raise RuntimeError('teardown')
def test_setup(bad_setup):
    pass
def test_teardown(bad_teardown):
    pass
@pytest.mark.skip(reason='synthetic skip')
def test_skip():
    pass
''',
                    'collection': "raise RuntimeError('collection failure')\n",
                    'crash': 'import os\ndef test_crash():\n    os._exit(23)\n',
                }[scenario]
                if scenario in ('transient', 'fail'):
                    source += "\ndef test_other():\n    from pathlib import Path\n    marker=Path(__file__).with_suffix('.other')\n    assert not marker.exists(), 'passing test must not run again'\n    marker.touch()\n"
                (tests / 'test_suite.py').write_text(source)
                (suite / '.gitignore').write_text('*.marker\n*.other\n*.ready\n*.json\n__pycache__/\n')
                for command in (['init'], ['add', '.'], ['-c', 'user.name=Test', '-c',
                                'user.email=test@example.invalid', 'commit', '-m', 'synthetic suite']):
                    subprocess.run(['git', *command], cwd=suite, check=True, capture_output=True)
                run = root / (scenario + '-run')
                env = dict(os.environ, PYTHONPATH=str(suite), PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_PLUGINS='xdist.plugin')
                env.pop('PYTEST_ADDOPTS', None)
                selection = []
                if scenario == 'transient':
                    selection = ['--', '-c', str(suite / 'setup.cfg'),
                                 '--rootdir', str(suite), 'tests/test_suite.py']
                elif scenario == 'fail':
                    selection = ['--', f'--inifile={suite / "setup.cfg"}', f'--rootdir={suite}',
                                 'tests/test_suite.py::test_fail', 'tests/test_suite.py::test_other']
                result = subprocess.run([
                    sys.executable, '-B', '-m', 'scripts.testing.block_runner',
                    '--run-dir', str(run), '--workers', '2' if scenario in ('pass', 'transient', 'fail') else '0',
                    *selection,
                ], cwd=launch, env=env,
                   capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=40)
                outcome = json.loads((run / 'outcome.json').read_text())
                first = json.loads((run / 'first/summary.json').read_text())
                self.assertEqual(result.returncode, 0 if scenario == 'pass' else 2 if scenario in ('collection', 'crash') else 1, result.stdout + result.stderr)
                self.assertEqual(outcome['complete'], scenario not in ('collection', 'crash'), first)
                self.assertEqual((run / 'rerun').exists(), scenario in ('transient', 'skip-rerun', 'xfail-rerun', 'fail', 'phases'))
                if scenario == 'pass':
                    tasks = [json.loads((tests / (task + '.json')).read_text()) for task in 'abcd']
                    self.assertEqual({task['task'] for task in tasks}, set('abcd'))
                    self.assertEqual({task['worker'] for task in tasks}, {'gw0', 'gw1'})
                    self.assertEqual(len({task['pid'] for task in tasks}), 2)
                    first_tasks = [min((task for task in tasks if task['worker'] == worker),
                                       key=lambda task: task['started']) for worker in ('gw0', 'gw1')]
                    self.assertLess(max(task['started'] for task in first_tasks),
                                    min(task['ended'] for task in first_tasks))
                    self.assertEqual(len(first['statuses']), 4)
                    self.assertEqual(set(first['statuses'].values()), {'passed'})
                if scenario == 'transient':
                    self.assertEqual(outcome['not_reproduced'], outcome['first_failed'])
                    self.assertTrue(outcome['first_failed'])
                if scenario in ('skip-rerun', 'xfail-rerun'):
                    self.assertTrue(outcome['first_failed'])
                    self.assertEqual(outcome['not_reproduced'], [])
                    second = json.loads((run / 'rerun/summary.json').read_text())
                    self.assertEqual(list(second['statuses'].values()),
                                     ['skipped' if scenario == 'skip-rerun' else 'xfail'])
                if scenario in ('fail', 'phases'):
                    self.assertEqual(runner.normalize_nodeids(suite, outcome['first_failed']),
                                     runner.normalize_nodeids(suite, outcome['rerun_failed']))
                if scenario == 'phases':
                    self.assertEqual(list(first['statuses'].values()).count('skipped'), 1)
                self.assertEqual(outcome['total_wall_seconds'],
                                 outcome['ended_monotonic'] - outcome['started_monotonic'])
                self.assertGreaterEqual(datetime.fromisoformat(outcome['ended_utc']),
                                        datetime.fromisoformat(outcome['started_utc']))
                rounds_wall = sum(json.loads(path.read_text())['wall_seconds']
                                  for path in run.glob('*/exit.json'))
                self.assertGreaterEqual(outcome['total_wall_seconds'], rounds_wall)
                start = json.loads((run / 'first/start.json').read_text())
                command = start['command']
                self.assertEqual(command[command.index('-n') + 1], '2' if scenario in ('pass', 'transient', 'fail') else '0')
                self.assertEqual(command.count('-n'), 1)
                self.assertEqual(command.count('-c'), 1)
                self.assertEqual(command[command.index('-c') + 1], str(suite.resolve() / 'setup.cfg'))
                self.assertEqual([arg for arg in command if arg.startswith('--rootdir=')], [f'--rootdir={suite.resolve()}'])
                self.assertNotIn('--checkpoint-log', command)
                self.assertIn(f'--checkpoint-log={run.resolve() / "first/checkpoint.jsonl"}', command)
                self.assertTrue(all(node.startswith('tests/') for node in first['collected']))
                self.assertTrue((run / 'first/junit.xml').exists() or scenario == 'crash')
                if (run / 'rerun').exists():
                    rerun = json.loads((run / 'rerun/start.json').read_text())['command']
                    self.assertTrue(any(arg.startswith('--checkpoint-failures=') for arg in rerun))
                    self.assertEqual(rerun.count('-c'), 1)
                    self.assertEqual(rerun[rerun.index('-c') + 1], str(suite.resolve() / 'setup.cfg'))
                    self.assertEqual([arg for arg in rerun if arg.startswith('--rootdir=')], [f'--rootdir={suite.resolve()}'])
                    self.assertEqual(rerun[rerun.index('-n') + 1], '0')
                    self.assertNotEqual(command[command.index('--basetemp') + 1], rerun[rerun.index('--basetemp') + 1])
                    second = json.loads((run / 'rerun/summary.json').read_text())
                    self.assertEqual(set(runner.normalize_nodeids(suite, second['collected'])),
                                     set(runner.normalize_nodeids(suite, first['failed'])))
                    self.assertEqual(command[command.index('-o') + 1], rerun[rerun.index('-o') + 1])
                if scenario in ('transient', 'fail'):
                    self.assertEqual(first['failed'], second['collected'])
                    self.assertEqual(first['failed'], [
                        'tests/test_suite.py::' + ('test_once' if scenario == 'transient' else 'test_fail')
                    ])
                    self.assertIn('tests/test_suite.py::test_other', first['collected'])
                    self.assertNotIn('tests/test_suite.py::test_other', second['collected'])

    def test_incomplete_and_corruption_matrix(self):
        node = 'tests/example.py::test_one'
        events = [{'event': 'collection', 'nodeids': [node]}]
        events += [dict(event='phase_report', nodeid=node, when=w, outcome='passed') for w in ('setup', 'call', 'teardown')]
        events += [dict(event='subtest_report', nodeid=node, when='call', outcome='failed')]
        events += [dict(event='node_finish', nodeid=node)]
        events += [dict(event='session_finish', exitstatus=1)]
        self.assertTrue(runner.summarize(events, 1)['complete'])
        self.assertEqual(runner.summarize(events, 1)['failed'], [node])
        for changed in (events[:-1], events[:2] + events[3:], events + [{'event': 'worker_error'}], events + [{'event': 'collection', 'nodeids': []}],
                        [e for e in events if e['event'] != 'node_finish'],
                        events + [dict(event='node_finish', nodeid=node)],
                        events + [dict(event='node_finish', nodeid='unknown')]):
            self.assertFalse(runner.summarize(changed, 1)['complete'])
        self.assertFalse(runner.summarize(events, 0)['complete'])
        for outcome, marker, expected in (('skipped', None, 'skipped'),
                                          ('skipped', 'expected', 'xfail'),
                                          ('passed', 'expected', 'xpass')):
            ordinary = [dict(e) for e in events if e['event'] != 'subtest_report']
            ordinary[-1]['exitstatus'] = 0
            ordinary[2].update(outcome=outcome, wasxfail=marker)
            summary = runner.summarize(ordinary, 0)
            self.assertTrue(summary['complete'], summary)
            self.assertEqual(summary['statuses'][node], expected)

    @unittest.skipIf(os.name == 'nt', 'LC_ALL=C ASCII parent encoding requires POSIX')
    def test_git_unicode_diff_in_non_utf8_parent(self):
        for changes_source in (False, True):
            with self.subTest(changes_source=changes_source), tempfile.TemporaryDirectory() as temp:
                parent = Path(temp)
                source = ('from pathlib import Path\ndef test_changes_source():\n'
                          '    Path("notes.txt").write_text("\\u4e2d\\u6587 second\\n", encoding="utf-8")\n'
                          '    assert False\n' if changes_source else 'def test_pass():\n    pass\n')
                suite = self._wall_suite(parent, source)
                (suite / '.gitignore').write_text('__pycache__/\n')
                notes = suite / 'notes.txt'
                notes.write_text('baseline\n', encoding='utf-8')
                for command in (['init'], ['add', '.'], ['-c', 'user.name=Test', '-c',
                                'user.email=test@example.invalid', 'commit', '-m', 'baseline']):
                    subprocess.run(['git', *command], cwd=suite, check=True, capture_output=True)
                notes.write_bytes('\u4e2d\u6587 first\r\n'.encode('utf-8'))
                command = ['git', 'diff', '--binary', '--no-ext-diff', '--no-textconv', 'HEAD', '--']
                raw_diff = subprocess.check_output(command, cwd=suite)
                display_diff = subprocess.check_output(command, cwd=suite, text=True,
                                                       encoding='utf-8', errors='replace').strip()
                run = parent / 'run'
                env = dict(os.environ, LC_ALL='C', LC_CTYPE='C', PYTHONUTF8='0', PYTHONCOERCECLOCALE='0',
                           PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_PLUGINS='xdist.plugin')
                env.pop('PYTEST_ADDOPTS', None)
                launcher = ('import json, locale; from scripts.testing import block_runner as r; '
                            'print(json.dumps(dict(parent_encoding=locale.getencoding())), flush=True); '
                            'raise SystemExit(r.main())')
                record = subprocess.run([
                    sys.executable, '-B', '-c', launcher, '--run-dir', str(run), '--workers', '0',
                    '--wall-timeout', '10', '--', 'tests'], cwd=suite, env=env, capture_output=True,
                    text=True, encoding='utf-8', errors='replace', timeout=30)
                encoding = json.loads(record.stdout.splitlines()[0])['parent_encoding']
                self.assertEqual('A'.encode(encoding), b'A')
                with self.assertRaises(UnicodeEncodeError):
                    '\u4e2d\u6587'.encode(encoding)
                self.assertEqual(record.returncode, 2 if changes_source else 0, record.stdout + record.stderr)
                version = json.loads((run / 'version.json').read_text(encoding='utf-8'))
                self.assertIn('\u4e2d\u6587 first', version['diff'])
                self.assertEqual(version['diff'], display_diff)
                self.assertEqual(version['diff_sha256'], hashlib.sha256(raw_diff).hexdigest())
                outcome = json.loads((run / 'outcome.json').read_text(encoding='utf-8'))
                self.assertEqual(outcome['complete'], not changes_source, outcome)
                if changes_source:
                    self.assertIn('source version changed before rerun', outcome['problems'])
                    self.assertFalse((run / 'rerun').exists())
                    final = json.loads((run / 'version-final.json').read_text(encoding='utf-8'))
                    self.assertNotEqual(version['diff_sha256'], final['diff_sha256'])
                else:
                    self.assertEqual(outcome['problems'], [])
                    first = json.loads((run / 'first/summary.json').read_text(encoding='utf-8'))
                    self.assertEqual(first['statuses'], {'tests/test_synthetic.py::test_pass': 'passed'})
                    self.assertEqual(version, json.loads((run / 'version-final.json').read_text(encoding='utf-8')))

    def test_git_source_identity_distinguishes_identical_lossy_displays(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            notes = source / 'notes.txt'
            notes.write_bytes(b'baseline\n')
            for command in (['init', '--object-format=sha1'], ['config', 'core.abbrev', '4'],
                            ['add', 'notes.txt'], ['-c', 'user.name=Test', '-c',
                             'user.email=test@example.invalid', 'commit', '-m', 'baseline']):
                subprocess.run(['git', *command], cwd=source, check=True, capture_output=True)
            versions = []
            diffs = []
            # Both real text blobs have SHA1 prefix fefa and decode to the same
            # replacement characters. Git's abbreviated index line is equal too.
            for content in (b'change \x81\xa4\n', b'change \x85\x86\n'):
                notes.write_bytes(content)
                raw = subprocess.check_output([
                    'git', 'diff', '--binary', '--no-ext-diff', '--no-textconv', 'HEAD', '--'], cwd=source)
                diffs.append(raw)
                versions.append(runner.source_version(source))
                self.assertEqual(versions[-1]['diff_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertNotEqual(diffs[0], diffs[1])
            self.assertEqual(diffs[0].decode('utf-8', errors='replace'), diffs[1].decode('utf-8', errors='replace'))
            self.assertEqual(versions[0]['diff'], versions[1]['diff'])
            self.assertNotEqual(versions[0], versions[1])

    def test_mock_process_argv_and_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'source'
            source.mkdir()
            for command in (['init'], ['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                                      'commit', '--allow-empty', '-m', 'baseline']):
                subprocess.run(['git', *command], cwd=source, check=True, capture_output=True)
            tracked = source / 'source.py'
            tracked.write_text('baseline\n')
            binary = source / 'fixture.bin'
            binary.write_bytes(b'\0baseline\xff')
            subprocess.run(['git', 'add', 'source.py', 'fixture.bin'],
                           cwd=source, check=True, capture_output=True)
            subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                            'commit', '-m', 'tracked'], cwd=source, check=True, capture_output=True)
            tracked.write_text('first dirty content\n')
            first_version = runner.source_version(source)
            tracked.write_text('second dirty content\n')
            second_version = runner.source_version(source)
            self.assertEqual(first_version['head'], second_version['head'])
            self.assertEqual(first_version['status'], second_version['status'])
            self.assertNotEqual(first_version['diff'], second_version['diff'])
            binary.write_bytes(b'\0first change\xff')
            first_binary_version = runner.source_version(source)
            binary.write_bytes(b'\0second change\xff')
            second_binary_version = runner.source_version(source)
            self.assertEqual(first_binary_version['head'], second_binary_version['head'])
            self.assertEqual(first_binary_version['status'], second_binary_version['status'])
            self.assertIn('GIT binary patch', first_binary_version['diff'])
            self.assertNotEqual(first_binary_version['diff'], second_binary_version['diff'])
            (source / 'untracked.py').write_text('untracked\n')
            self.assertIn('?? untracked.py', runner.source_version(source)['status'])
            untracked = source / "test new [case]'s.py"
            versions = []
            for index, value in enumerate(('False', 'True')):
                untracked.write_text(f'def test_new():\n    assert {value}\n')
                versions.append(runner.source_version(source))
                self.assertEqual(versions[-1]['untracked'], [untracked.name, 'untracked.py'])
                run = Path(temp) / f'untracked-{index}'
                with patch.object(runner, '__file__', str(source / 'scripts/testing/block_runner.py')), patch.object(
                    runner, 'execute'
                ) as execution:
                    self.assertEqual(runner.main(['--run-dir', str(run)]), 2)
                    execution.assert_not_called()
                self.assertEqual(json.loads((run / 'version.json').read_text()), versions[-1])
                outcome = json.loads((run / 'outcome.json').read_text())
                self.assertFalse(outcome['complete'])
                self.assertIn('untracked files present', outcome['problems'][0])
                self.assertFalse((run / 'first').exists())
            self.assertEqual(versions[0], versions[1])

            existing = Path(temp) / 'existing'
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                runner.main(['--run-dir', str(existing)])
            for flag in ('-n2', '--dist=load', '-x', '--maxfail=1', '--lf', '--collect-only', '-ofoo=bar', '--checkpoint-failures=x',
                         f'-c{source / "setup.cfg"}', f'--inifile={source / "setup.cfg"}',
                         f'--rootdir={source}', '-c', '--inifile', '--rootdir'):
                with self.assertRaises(SystemExit):
                    runner.main(['--run-dir', str(existing / 'new'), '--', flag])
            self.assertFalse((existing / 'new').exists())

            for options in (['--workers', '-1'], ['--workers', '17'], ['--dist', 'each']):
                with self.assertRaises(SystemExit):
                    runner.main(['--run-dir', str(existing / 'new'), *options])
            node = 'test_suite.py::test_fail'
            (source / 'test_suite.py').write_text('def test_fail():\n    assert False\n')
            failed = dict(complete=True, failed=[node], collected=[node], statuses={node: 'failed'}, exitcode=1)
            passed = dict(complete=True, failed=[], collected=[node], statuses={node: 'passed'}, exitcode=0)
            subprocess.run(['git', 'add', '--', untracked.name, 'untracked.py', 'test_suite.py'],
                           cwd=source, check=True, capture_output=True)
            with patch.object(runner, '__file__', str(source / 'scripts/testing/block_runner.py')), patch.object(
                runner, 'execute', return_value=passed
            ) as execution:
                self.assertEqual(runner.main(['--run-dir', str(Path(temp) / 'tracked-dirty')]), 0)
                execution.assert_called_once()
                self.assertEqual(execution.call_args.args[4], 6)
                self.assertEqual(execution.call_args.args[5], 'worksteal')
            with patch.object(runner, '__file__', str(source / 'scripts/testing/block_runner.py')), patch.object(
                runner, 'execute', return_value=passed
            ) as execution:
                self.assertEqual(runner.main(['--run-dir', str(Path(temp) / 'sixteen-workers'), '--workers', '16']), 0)
                execution.assert_called_once()
                self.assertEqual(execution.call_args.args[4], 16)
            baseline = dict(head='baseline', diff='', status='', untracked=[])
            changed = dict(baseline, diff=' M source.py')
            for index, unavailable in enumerate((dict(baseline, head=None),
                                                 dict(baseline, untracked=None))):
                run = Path(temp) / f'unavailable-{index}'
                with patch.object(runner, 'source_version', return_value=unavailable), patch.object(
                    runner, 'execute'
                ) as execution:
                    self.assertEqual(runner.main(['--run-dir', str(run)]), 2)
                    execution.assert_not_called()
                outcome = json.loads((run / 'outcome.json').read_text())
                self.assertFalse(outcome['complete'])
                self.assertEqual(outcome['problems'], ['source version unavailable before first round'])
            with patch.object(runner.subprocess, 'run', side_effect=FileNotFoundError('git')):
                self.assertTrue(all(value is None for value in runner.source_version(source).values()))
            for phase in ('before-rerun', 'final'):
                run = existing / phase
                versions = [baseline, changed, changed] if phase == 'before-rerun' else [baseline, baseline, changed]
                with patch.object(runner, '__file__', str(source / 'scripts/testing/block_runner.py')), patch.object(runner, 'source_version', side_effect=versions), patch.object(
                    runner, 'execute', side_effect=[failed, passed]
                ) as execution:
                    self.assertEqual(runner.main(['--run-dir', str(run), '--workers', '0']), 2)
                outcome = json.loads((run / 'outcome.json').read_text())
                self.assertFalse(outcome['complete'])
                self.assertEqual(outcome['not_reproduced'], [])
                self.assertTrue(outcome['problems'])
                self.assertEqual(execution.call_count, 1 if phase == 'before-rerun' else 2)

            self.assertEqual(runner.normalize_nodeids(source, [source.name + '/' + node]), [node])
            nested = source / source.name
            nested.mkdir()
            (nested / 'test_suite.py').write_text('def test_fail():\n    pass\n')
            for invalid in ([source.name + '/' + node], ['missing.py::test_fail'], [1],
                            [node, source.as_posix() + '/' + node],
                            [(Path(temp) / 'unrelated.py').as_posix() + '::test_fail']):
                with self.assertRaises(ValueError):
                    runner.normalize_nodeids(source, invalid)
                with patch.object(runner, '__file__', str(source / 'scripts/testing/block_runner.py')), patch.object(
                    runner, 'source_version', return_value=baseline
                ), patch.object(runner, 'execute', return_value=dict(failed, failed=invalid)) as execution:
                    run = Path(temp) / ('invalid-' + str(len(list(Path(temp).glob('invalid-*')))))
                    self.assertEqual(runner.main(['--run-dir', str(run)]), 2)
                    execution.assert_called_once()
                    self.assertFalse((run / 'rerun').exists())
