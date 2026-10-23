"""Run standard pytest/xdist, then one serial explicit-failure confirmation.

Acceptance inputs must be Git-tracked or match an explicit SHA256 source
manifest; tracked uncommitted changes are allowed in Git mode.
Each pytest invocation has a wall-timeout budget for waiting. Process
observation and reaping depend on the OS; cleanup and source checks add time.
POSIX cleanup targets this runner's new session group; Windows uses an owned
Job Object containing pytest and its descendants.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def normalize_nodeids(root, nodeids):
    """Resolve only unique real test files inside this exact source checkout.

    Workers may report paths relative to an ancestor of the checkout when an
    existing checkpoint option value participates in pytest's root discovery.
    Resolve against actual ancestors, never strip an arbitrary nodeid prefix.
    """
    root = root.resolve(strict=True)
    normalized = []
    for node in nodeids:
        if not isinstance(node, str):
            raise ValueError(f'invalid test nodeid: {node!r}')
        filename, separator, test = node.partition('::')
        if not separator or not filename or not test:
            raise ValueError(f'invalid test nodeid: {node!r}')
        candidates = set()
        for base in (root, *root.parents):
            path = (base / filename).resolve()
            if path.is_relative_to(root) and path.is_file():
                candidates.add(path)
        if len(candidates) != 1:
            raise ValueError(f'test path is missing or ambiguous in source root: {node!r}')
        path = candidates.pop()
        normalized.append(path.relative_to(root).as_posix() + '::' + test)
    if len(set(normalized)) != len(normalized):
        raise ValueError('duplicate normalized test nodeids')
    return normalized


def summarize(events, exitcode):
    collections = [e['nodeids'] for e in events if e['event'] == 'collection']
    nodes = collections[0] if collections else []
    problems = []
    if not collections or any(c != nodes for c in collections):
        problems.append('missing or inconsistent collection')
    if len(nodes) != len(set(nodes)):
        problems.append('duplicate collected nodeids')
    finishes = [e['exitstatus'] for e in events if e['event'] == 'session_finish']
    if finishes != [exitcode] or exitcode not in (0, 1, 5):
        problems.append('missing or abnormal session finish')
    if any(e['event'] in ('collection_error', 'worker_error', 'internal_error', 'selection_error') for e in events):
        problems.append('collection, worker, internal or selection error')
    reports = {n: [] for n in nodes}
    for e in events:
        if e['event'] in ('phase_report', 'subtest_report'):
            if e['nodeid'] not in reports:
                problems.append('uncollected report')
            else:
                reports[e['nodeid']].append(e)
    endings = Counter(e['nodeid'] for e in events if e['event'] == 'node_finish')
    if set(endings) - set(nodes):
        problems.append('uncollected node finish')
    missing, failed, statuses = [], [], {}
    for node, details in reports.items():
        phases = [e for e in details if e['event'] == 'phase_report']
        order = [e['when'] for e in phases]
        expected = ['setup', 'call', 'teardown']
        if phases and phases[0]['when'] == 'setup' and phases[0]['outcome'] != 'passed':
            expected = ['setup', 'teardown']
        if order != expected or endings[node] != 1:
            missing.append(node)
        is_failed = any(e['outcome'] == 'failed' for e in details)
        if is_failed:
            failed.append(node)
        statuses[node] = ('incomplete' if node in missing else 'failed' if is_failed else
                          'xfail' if any(e['outcome'] == 'skipped' and e.get('wasxfail') for e in phases) else
                          'skipped' if any(e['outcome'] == 'skipped' for e in phases) else
                          'xpass' if any(e.get('wasxfail') for e in phases) else 'passed')
    if missing:
        problems.append('missing or invalid phases or node finish')
    if (exitcode == 1) != bool(failed):
        problems.append('exit code and failure reports disagree')
    if exitcode == 5 and nodes:
        problems.append('no-tests exit with collected nodes')
    return dict(complete=not problems, collected=nodes, failed=failed, missing=missing,
                statuses=statuses, problems=problems)


def checkout_args(root, args):
    """Accept matching legacy root/config flags and reject other checkouts."""
    root = root.resolve()
    config = root / 'setup.cfg'
    remaining = []
    arguments = iter(args)
    for arg in arguments:
        option, equals, value = arg.partition('=')
        if option in ('-c', '--inifile', '--rootdir'):
            if not equals:
                try:
                    value = next(arguments)
                except StopIteration:
                    raise ValueError(f'{option} requires a path') from None
        elif arg.startswith('-c'):
            option, value = '-c', arg[2:]
        else:
            remaining.append(arg)
            continue
        expected = root if option == '--rootdir' else config
        if not value or (root / value).resolve() != expected:
            raise ValueError(f'{option} must use this checkout: {expected}')
    return remaining


def group_members(process, known):
    """Check only our new session, retaining a live member identity as anchor.

    Read before reaping the leader. Once the group is empty it is never signaled
    again; an unrecognized replacement group is not ours to terminate.
    """
    record = subprocess.run(['/bin/ps', '-axo', 'pid=,pgid=,stat=,lstart='],
                            capture_output=True, text=True, check=True, timeout=2)
    members = {}
    for line in record.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and parts[1] == str(process.pid):
            members[int(parts[0])] = (parts[3].strip(), parts[2])
    if members:
        if not known:
            if (process.pid not in members or os.getpgid(process.pid) != process.pid
                    or os.getsid(process.pid) != process.pid):
                raise RuntimeError('owned process-group identity unavailable')
        elif (not any(known.get(pid) == birth for pid, (birth, _) in members.items())
              or any(pid in known and known[pid] != birth for pid, (birth, _) in members.items())):
            raise RuntimeError('owned process-group identity changed')
        known.update((pid, birth) for pid, (birth, _) in members.items())
    return members


def wait_owned(process, deadline, interruption, known):
    while True:
        if interruption and interruption[0] is not None:
            return 'interrupted'
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 'wall_timeout'
        if os.name == 'posix':
            # Popen.poll()/wait() would reap a fast-exiting leader before its
            # descendants are identified. ps observes exit without reaping it.
            record = subprocess.run(['/bin/ps', '-p', str(process.pid), '-o', 'stat='],
                                    capture_output=True, text=True, timeout=min(2, remaining))
            if interruption and interruption[0] is not None and record.returncode == -interruption[0]:
                return 'interrupted'
            if record.returncode not in (0, 1):
                raise RuntimeError('cannot observe owned child status')
            if not record.stdout.strip() or 'Z' in record.stdout:
                group_members(process, known)
                if process.poll() is None:
                    raise RuntimeError('owned child disappeared without an exit status')
                return 'completed'
            time.sleep(min(.25, max(0, deadline - time.monotonic())))
        else:
            try:
                process.wait(timeout=min(.25, remaining))
                return 'completed'
            except subprocess.TimeoutExpired:
                pass


def windows_job(process):
    """An owned kill-on-close job prevents an interrupted SSH parent orphaning its worker."""
    import ctypes
    from ctypes import wintypes
    class Basic(ctypes.Structure):
        _fields_=[('time1',ctypes.c_int64),('time2',ctypes.c_int64),('flags',wintypes.DWORD),
                  ('min',ctypes.c_size_t),('max',ctypes.c_size_t),('active',wintypes.DWORD),
                  ('affinity',ctypes.c_size_t),('priority',wintypes.DWORD),('scheduling',wintypes.DWORD)]
    class IO(ctypes.Structure):
        _fields_=[(f'v{i}',ctypes.c_uint64) for i in range(6)]
    class Extended(ctypes.Structure):
        _fields_=[('basic',Basic),('io',IO),('process',ctypes.c_size_t),('job',ctypes.c_size_t),
                  ('peak1',ctypes.c_size_t),('peak2',ctypes.c_size_t)]
    k = ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateJobObjectW.restype=wintypes.HANDLE
    k.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR]
    k.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
    k.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
    k.TerminateJobObject.argtypes=[wintypes.HANDLE,wintypes.UINT]
    k.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=k.CreateJobObjectW(None,None)
    if not handle:
        raise OSError(ctypes.get_last_error(),'CreateJobObject failed')
    info=Extended();info.basic.flags=0x2000
    if not k.SetInformationJobObject(handle,9,ctypes.byref(info),ctypes.sizeof(info)) or not k.AssignProcessToJobObject(handle,process._handle):
        error=ctypes.get_last_error();k.CloseHandle(handle)
        raise OSError(error,'owned worker job setup failed')
    return k,handle

def windows_active(job):
    import ctypes
    from ctypes import wintypes

    class Accounting(ctypes.Structure):
        _fields_ = [(name, ctypes.c_int64) for name in ('user', 'kernel', 'period_user', 'period_kernel')] + [
            (name, wintypes.DWORD) for name in ('faults', 'total', 'active', 'terminated')]

    kernel, handle = job
    kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                wintypes.DWORD, ctypes.c_void_p]
    info = Accounting()
    if not kernel.QueryInformationJobObject(handle, 1, ctypes.byref(info), ctypes.sizeof(info), None):
        raise OSError('owned job process query failed')
    return info.active


def cleanup_owned(process, known, *, job=None):
    scope = 'windows_job' if job is not None else 'posix_session_group' if os.name == 'posix' else 'direct_child'
    result = dict(scope=scope, complete=False, signals=[], remaining=[], problems=[])
    try:
        if job is not None:
            kernel, handle = job
            result.update(remaining=None, active_processes=None)
            try:
                active = result['active_processes'] = windows_active(job)
                process.poll()
                result['residual_after_leader_exit'] = process.returncode is not None and bool(active)
                if active:
                    if not kernel.TerminateJobObject(handle, 1):
                        raise OSError('owned job termination failed')
                    result['signals'].append('TerminateJobObject')
                    until = time.monotonic() + 5
                    while active and time.monotonic() < until:
                        process.poll()
                        active = result['active_processes'] = windows_active(job)
                        if active:
                            time.sleep(.05)
                result['remaining'] = [] if not active else None
                if not active:
                    process.wait(timeout=5)
                result['complete'] = not active and process.returncode is not None
            finally:
                if not kernel.CloseHandle(handle):
                    result['complete'] = False
                    result['problems'].append('owned job handle close failed')
        elif os.name == 'posix':
            if not known:
                raise RuntimeError('owned process-group identity unavailable; no signals sent')
            members = group_members(process, known)
            # Capture descendants while the owned leader still anchors the
            # group, then reap it before attempting any signal (Mac sandbox).
            process.poll()
            members = group_members(process, known) if members else {}
            result['residual_after_leader_exit'] = process.returncode is not None and bool(members)
            for signum in (signal.SIGTERM, signal.SIGKILL):
                if not members:
                    break
                os.killpg(process.pid, signum)
                result['signals'].append(signal.Signals(signum).name)
                until = time.monotonic() + 5
                while members and time.monotonic() < until:
                    process.poll()
                    members = group_members(process, known)
                    if members:
                        time.sleep(.05)
            result['remaining'] = sorted(members)
            process.poll()
            result['complete'] = not members and process.returncode is not None
        else:
            # On Windows this is only the still-gated startup child when job
            # creation/assignment failed; pytest has never been released.
            result['residual_after_leader_exit'] = False
            for name, method in (('terminate', process.terminate), ('kill', process.kill)):
                if process.poll() is not None:
                    break
                method()
                result['signals'].append(name)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            result['complete'] = process.returncode is not None
            result['remaining'] = [] if result['complete'] else [process.pid]
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        # Never widen the target or retry with another signaling mechanism.
        result['problems'].append(str(exc))
    if not result['complete']:
        result['problems'].append('owned cleanup incomplete')
    return result


def execute(root, directory, cache, args, workers, dist, failures=None,
            *, wall_timeout=7200, interruption=None, sources=None):
    args = checkout_args(root, args)
    directory.mkdir()
    command = [sys.executable, '-B', '-m', 'pytest', '-n', str(workers),
               '-c', str(root / 'setup.cfg'), f'--rootdir={root}']
    if workers:
        command += ['--dist', dist]
    command += ['-p', 'scripts.testing.pytest_checkpoint',
                f'--checkpoint-log={directory / "checkpoint.jsonl"}', '--junitxml', str(directory / 'junit.xml'),
                '--basetemp', str(directory / 'temp'), '-o', f'cache_dir={cache}',
                '--durations=0', '--durations-min=0', '-ra']
    if failures is not None:
        command += [f'--checkpoint-failures={failures}']
    if sources is not None:
        command += [f'--checkpoint-sources={sources}']
    command += args
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', PYTEST_ADDOPTS='')
    env['PYTHONPATH'] = os.pathsep.join([str(root), str(root / 'src'), str(root / 'tests'),
                                       env.get('PYTHONPATH', '')])
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    process = None
    job = None
    known = {}
    reason = 'completed'
    timed_out = False
    problems = []
    cleanup = dict(scope='not_started', complete=True, signals=[], remaining=[], problems=[])
    with (directory / 'stdout.log').open('wb') as stdout, (directory / 'stderr.log').open('wb') as stderr:
        try:
            if interruption and interruption[0] is not None:
                reason = 'interrupted'
            else:
                launch_command = command
                if os.name == 'nt':
                    # EOF is refusal, never permission. No pytest import or
                    # worker creation can precede successful job assignment.
                    gate = ("import sys\n"
                            "if sys.stdin.buffer.read(1) != b'1': raise SystemExit(2)\n"
                            "sys.stdin.close()\n"
                            "import os, runpy\n"
                            "sys.stdin = open(os.devnull)\n"
                            "sys.argv = ['pytest', *sys.argv[1:]]\n"
                            "runpy.run_module('pytest', run_name='__main__')\n")
                    launch_command = [sys.executable, '-B', '-c', gate, *command[4:]]
                    # Match multiprocessing's Windows venv launch: its native
                    # redirector can spawn Python before our job is assigned.
                    if os.path.normcase(sys.executable) != os.path.normcase(sys._base_executable):
                        launch_command[0] = sys._base_executable
                        env['__PYVENV_LAUNCHER__'] = sys.executable
                process = subprocess.Popen(launch_command, cwd=root, env=env,
                                           stdin=subprocess.PIPE if os.name == 'nt' else subprocess.DEVNULL,
                                           stdout=stdout, stderr=stderr, start_new_session=os.name == 'posix')
                save(directory / 'start.json', dict(command=command, launch_command=launch_command,
                     cwd=str(root), pid=process.pid,
                     pgid=process.pid if os.name == 'posix' else None,
                     cleanup_scope='posix_session_group' if os.name == 'posix' else 'windows_job' if os.name == 'nt' else 'direct_child',
                     wall_timeout_seconds=wall_timeout, timeout_scope='one_pytest_invocation',
                     started_monotonic=started, started_utc=started_utc))
                if os.name == 'posix':
                    if not group_members(process, known):
                        raise RuntimeError('owned process-group identity unavailable at start')
                elif os.name == 'nt':
                    job = windows_job(process)
                    process.stdin.write(b'1')
                    process.stdin.close()
                reason = wait_owned(process, started + wall_timeout, interruption, known)
        except KeyboardInterrupt:
            reason = 'interrupted'
        except subprocess.TimeoutExpired as exc:
            reason = 'wall_timeout' if time.monotonic() - started >= wall_timeout else 'execution_error'
            problems.append(str(exc))
        except Exception as exc:
            reason = 'execution_error'
            problems.append(str(exc))
            # A failed observation cannot authorize signaling the group later.
            known.clear()
        finally:
            timed_out = reason == 'wall_timeout'
            if process is not None:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except OSError as exc:
                        reason = 'execution_error'
                        problems.append(str(exc))
                cleanup = cleanup_owned(process, known, job=job)
    code = process.returncode if process is not None else None
    if not cleanup['complete']:
        reason = 'cleanup_incomplete'
    elif reason == 'completed' and cleanup.get('residual_after_leader_exit'):
        reason = 'residual_descendants'
    # Preserve the timeout cause even when cleanup itself fails.
    if reason != 'completed':
        problems.append(reason)
    problems.extend(cleanup['problems'])
    ended = time.monotonic()
    save(directory / 'exit.json', dict(exitcode=code, started_monotonic=started,
                                     ended_monotonic=ended, wall_seconds=ended - started,
                                     termination_reason=reason, timed_out=timed_out, cleanup=cleanup,
                                     ended_utc=datetime.now(timezone.utc).isoformat()))
    try:
        events = [json.loads(line) for line in (directory / 'checkpoint.jsonl').read_text(encoding='utf-8').splitlines()]
        result = summarize(events, code)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = dict(complete=False, failed=[], collected=[], problems=[f'invalid evidence: {exc}'])
    if not (directory / 'junit.xml').is_file():
        result['complete'] = False
        result['problems'].append('missing JUnit')
    if problems:
        result['complete'] = False
        result['problems'].extend(problems)
    result['exitcode'] = code
    result.update(termination_reason=reason, timed_out=timed_out, cleanup=cleanup)
    save(directory / 'summary.json', result)
    return result


_OUTPUT_ROOTS = ('.git/', '.codegraph/', '.venv/', 'data/', 'quant_data/', 'logs/',
                 'reports/', 'output/', 'tmp/', 'quant_workflow/reports/', 'quant_workflow/mlruns/',
                 'quant_workflow/scratch/', 'quant_workflow/scripts/analysis_reports/',
                 'quant_workflow/scripts/backtest_results/')
_SOURCE_ROOTS = ('src', 'api', 'bot', 'data_provider', 'scripts', 'tests',
                 'quant_workflow/scripts', 'quant_workflow/tests')
_SOURCE_SUFFIXES = {'.py', '.sh', '.ps1', '.bat', '.cmd', '.cfg', '.ini', '.toml', '.yaml', '.yml',
                    '.json', '.txt', '.js', '.jsx', '.mjs', '.cjs', '.ts', '.tsx', '.html', '.css', '.sql'}


def manifest_version(root, manifest):
    """Verify an explicit source allowlist, without traversing data/output trees."""
    root = root.resolve()
    version = dict(mode='source_manifest', source_root=str(root), files={}, problems=[])
    try:
        raw = Path(manifest).read_bytes()
        document = json.loads(raw)
        declared = document['files']
        if not isinstance(declared, dict) or not declared:
            raise ValueError('source manifest files must be a nonempty mapping')
        version.update(manifest_sha256=hashlib.sha256(raw).hexdigest(),
                       declared_base_commit=document.get('base_commit'))
        for name, expected in declared.items():
            path = Path(name)
            if (not name or '\\' in name or ':' in name or path.is_absolute()
                    or '..' in path.parts or path.as_posix() != name):
                raise ValueError(f'invalid source manifest path: {name!r}')
            link = expected.get('mode') == '120000'
            test_input = name.startswith(('tests/', 'quant_workflow/tests/')) and '__pycache__' not in path.parts
            if path.suffix not in _SOURCE_SUFFIXES and not link and not test_input:
                continue
            if name.lower().startswith(_OUTPUT_ROOTS):
                raise ValueError(f'output/data path in source manifest: {name}')
            actual = root / path
            if not actual.resolve(strict=True).is_relative_to(root):
                raise ValueError(f'source manifest path escapes checkout: {name}')
            if any(parent.is_symlink() for parent in actual.parents if parent != root and parent.is_relative_to(root)):
                raise ValueError(f'source manifest path traverses a symlink: {name}')
            if link:
                if not actual.is_symlink():
                    raise ValueError(f'declared source symlink is not a symlink: {name}')
                target = os.readlink(actual)
                if Path(target).is_absolute():
                    raise ValueError(f'source symlink target must be relative: {name}')
                content = os.fsencode(target)
            else:
                if actual.is_symlink():
                    raise ValueError(f'undeclared source symlink: {name}')
                content = actual.read_bytes()
            observed = dict(sha256=hashlib.sha256(content).hexdigest(), size=len(content))
            if link:
                observed.update(mode='120000', target=target)
            version['files'][name] = observed
            if observed['sha256'] != expected.get('sha256') or observed['size'] != expected.get('size'):
                version['problems'].append(f'source manifest content mismatch: {name}')
        required = {'setup.cfg', 'scripts/testing/block_runner.py', 'scripts/testing/pytest_checkpoint.py'}
        for source_root in _SOURCE_ROOTS:
            for directory, children, files in os.walk(root / source_root, followlinks=False):
                base = Path(directory)
                for name in children:
                    path = base / name
                    if path.is_symlink() and path.relative_to(root).as_posix() not in version['files']:
                        version['problems'].append(f'unlisted source directory symlink: {path.relative_to(root).as_posix()}')
                children[:] = [name for name in children if name != '__pycache__' and not
                               ((base / name).relative_to(root).as_posix().lower() + '/').startswith(_OUTPUT_ROOTS)]
                required.update((base / name).relative_to(root).as_posix() for name in files if name.endswith('.py'))
        required.update(path.relative_to(root).as_posix() for path in root.glob('*.py'))
        required.update(path.relative_to(root).as_posix() for path in (root / 'quant_workflow').glob('*.py'))
        for name in sorted(required - version['files'].keys()):
            version['problems'].append(f'source absent from source manifest: {name}')
        for name in document.get('missing', []):
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError(f'invalid missing source path: {name!r}')
            if (root / name).exists():
                version['problems'].append(f'declared missing source appeared: {name}')
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        version['problems'].append(f'invalid source manifest: {exc}')
    version['complete'] = not version['problems']
    return version


def source_version(root, source_manifest=None):
    if source_manifest is not None:
        return manifest_version(root, source_manifest)
    version = {}
    for name, command in [('head', ['rev-parse', 'HEAD']),
                          ('diff', ['diff', '--binary', '--no-ext-diff', '--no-textconv', 'HEAD', '--']),
                          ('status', ['status', '--short'])]:
        try:
            record = subprocess.run(['git', *command], cwd=root, capture_output=True, text=True)
            version[name] = record.stdout.strip() if record.returncode == 0 else None
        except OSError:
            version[name] = None
    try:
        record = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard', '-z'],
                                cwd=root, capture_output=True)
        version['untracked'] = ([os.fsdecode(path) for path in record.stdout.split(b'\0') if path]
                                if record.returncode == 0 else None)
    except OSError:
        version['untracked'] = None
    return version


def _main(argv, interruption):
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--source-manifest', type=Path,
                        help='Explicit source SHA256 allowlist; verifies untracked code while excluding generated data.')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--wall-timeout', type=float, default=7200,
                        help='Wall-timeout budget in seconds per pytest invocation (default: 7200); process observation/reaping depend on the OS; cleanup and source checks add time. Not a total task deadline.')
    parser.add_argument('--dist', choices=('worksteal', 'loadscope', 'load', 'loadfile'), default='worksteal')
    options, args = parser.parse_known_args(argv)
    if args[:1] == ['--']:
        args = args[1:]
    # Execution/evidence/selection controls belong to this entry, not passthrough.
    forbidden = ('-n', '--numprocesses', '--dist', '--checkpoint-log', '--checkpoint-failures', '--junitxml',
                 '--basetemp', '-o', '--override-ini', '--lf', '--last-failed', '--lfnf',
                 '-x', '--exitfirst', '--maxfail', '--collect-only', '--co', '--stepwise', '--sw', '--wall-timeout',
                 '--source-manifest', '--checkpoint-sources')
    if not math.isfinite(options.wall_timeout) or options.wall_timeout <= 0:
        parser.error('wall-timeout must be a positive finite number of seconds')
    if not 0 <= options.workers <= 16:
        parser.error('workers must be between 0 and 16')
    if any(a == p or a.startswith(p + '=') or
                                (p in ('-n', '-o') and a.startswith(p)) for a in args for p in forbidden):
        parser.error('passthrough cannot override execution, selection or evidence controls')
    root = Path(__file__).resolve().parents[2]
    try:
        args = checkout_args(root, args)
    except ValueError as exc:
        parser.error(str(exc))
    run = options.run_dir.resolve()
    run.mkdir(parents=True)  # Refuse to overwrite a previous run.
    cache = run / 'cache'
    version_args = (root, options.source_manifest) if options.source_manifest else (root,)
    version = source_version(*version_args)
    save(run / 'version.json', version)
    first = second = None
    problems = []
    if options.source_manifest:
        if not version.get('complete'):
            problems.extend(version.get('problems', ['source manifest unavailable']))
    elif any(version.get(key) is None for key in ('head', 'diff', 'status', 'untracked')):
        problems.append('source version unavailable before first round')
    if version.get('untracked'):
        problems.append('untracked files present; add acceptance inputs to Git before running')
    if interruption[0] is not None:
        problems.append(f'runner interrupted by signal {interruption[0]}')
    if not problems:
        first = execute(root, run / 'first', cache, args, options.workers, options.dist,
                        wall_timeout=options.wall_timeout, interruption=interruption,
                        sources=run / 'version.json' if options.source_manifest else None)
    if first is not None and first['complete'] and first['failed']:
        before_rerun = source_version(*version_args)
        save(run / 'version-before-rerun.json', before_rerun)
        if before_rerun != version:
            problems.append('source version changed before rerun')
    if interruption[0] is not None and not problems:
        problems.append(f'runner interrupted by signal {interruption[0]}')
    if first is not None and first['complete'] and first['failed'] and not problems:
        try:
            normalized = normalize_nodeids(root, first['failed'])
            failures = run / 'failures.json'
            save(failures, dict(source_root=str(root), first_failed=first['failed'], nodeids=normalized))
        except (OSError, ValueError) as exc:
            problems.append(f'invalid first failure selection: {exc}')
        else:
            second = execute(root, run / 'rerun', cache, args, 0, options.dist, failures=failures,
                             wall_timeout=options.wall_timeout, interruption=interruption,
                             sources=run / 'version.json' if options.source_manifest else None)
            # The plugin also checks this before pytest starts any test body.
            try:
                selected = normalize_nodeids(root, second['collected'])
                second['normalized_statuses'] = dict(zip(selected, (second['statuses'][n] for n in second['collected'])))
                if set(selected) != set(normalized):
                    raise ValueError('rerun selection differs from first failures')
            except (OSError, ValueError, KeyError) as exc:
                second['complete'] = False
                second['problems'].append(str(exc))
    if first is not None:
        final_version = source_version(*version_args)
        save(run / 'version-final.json', final_version)
        if final_version != version:
            problems.append('source version changed during run')
    if interruption[0] is not None and not problems:
        problems.append(f'runner interrupted by signal {interruption[0]}')
    complete = not problems and first is not None and first['complete'] and (second is None or second['complete'])
    first_failed = first['failed'] if first else []
    outcome = dict(complete=complete, first_failed=first_failed,
                   rerun_failed=second['failed'] if second else [],
                   not_reproduced=[n for n, canonical in zip(first_failed, normalized if second else [])
                                   if complete and second and second['normalized_statuses'].get(canonical) == 'passed'],
                   problems=problems, wall_timeout_per_round=options.wall_timeout,
                   round_problems={name: round_result.get('problems', []) for name, round_result in
                                   (('first', first), ('rerun', second)) if round_result is not None and not round_result['complete']},
                   started_monotonic=started,
                   ended_monotonic=time.monotonic(), started_utc=started_utc,
                   ended_utc=datetime.now(timezone.utc).isoformat())
    outcome['total_wall_seconds'] = outcome['ended_monotonic'] - started
    save(run / 'outcome.json', outcome)
    print(json.dumps(outcome, ensure_ascii=False))
    return 2 if not complete else 1 if first['failed'] else first['exitcode']


def main(argv=None):
    interruption = [None]
    previous = signal.signal(signal.SIGTERM, lambda signum, frame: interruption.__setitem__(0, signum))
    try:
        return _main(argv, interruption)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    sys.exit(main())
