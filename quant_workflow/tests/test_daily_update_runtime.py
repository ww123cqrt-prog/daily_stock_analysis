import importlib.util
import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "_daily_update_runtime.py"
    spec = importlib.util.spec_from_file_location("_daily_update_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DailyUpdateRuntimeTest(unittest.TestCase):
    def test_live_process_lock_is_reported_as_controlled_skip(self):
        module = load_module()
        with TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "daily_update.lock.json"
            lock_path.write_text(
                json.dumps({"pid": 4242, "run_id": "existing", "started_at": "2026-07-24T18:40:00+08:00"}),
                encoding="utf-8",
            )
            with patch.object(module, "pid_is_alive", return_value=True):
                with self.assertRaises(module.LockHeldError) as raised:
                    module.DailyUpdateLock(lock_path).acquire("new-run")
        self.assertEqual(raised.exception.owner["run_id"], "existing")

    def test_dead_process_lock_is_reclaimed_and_owned_release_removes_it(self):
        module = load_module()
        with TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "daily_update.lock.json"
            lock_path.write_text(
                json.dumps({"pid": 4242, "run_id": "stale", "started_at": "2026-07-23T18:40:00+08:00"}),
                encoding="utf-8",
            )
            with patch.object(module, "pid_is_alive", return_value=False):
                lock = module.DailyUpdateLock(lock_path)
                owner = lock.acquire("new-run")
                self.assertEqual(owner["pid"], os.getpid())
                self.assertEqual(owner["run_id"], "new-run")
                lock.release()
            self.assertFalse(lock_path.exists())

    def test_recent_malformed_lock_is_not_silently_deleted(self):
        module = load_module()
        with TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "daily_update.lock.json"
            lock_path.write_text("not-json", encoding="utf-8")
            with self.assertRaises(module.LockHeldError):
                module.DailyUpdateLock(lock_path, malformed_stale_after_seconds=3600).acquire("new-run")
            self.assertTrue(lock_path.exists())

    def test_expired_malformed_lock_is_reclaimed(self):
        module = load_module()
        with TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "daily_update.lock.json"
            lock_path.write_text("not-json", encoding="utf-8")
            old = time.time() - 7200
            os.utime(lock_path, (old, old))
            lock = module.DailyUpdateLock(lock_path, malformed_stale_after_seconds=3600)
            owner = lock.acquire("new-run")
            self.assertEqual(owner["run_id"], "new-run")
            lock.release()

    def test_atomic_status_write_leaves_parseable_latest_json(self):
        module = load_module()
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "runtime" / "status" / "latest.json"
            module.write_json_atomic(path, {"status": "running", "run_id": "r1"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "running")
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
