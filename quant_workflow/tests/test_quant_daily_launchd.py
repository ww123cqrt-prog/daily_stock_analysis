import plistlib
import unittest
from pathlib import Path


class QuantDailyLaunchdTest(unittest.TestCase):
    def test_template_is_parseable_and_keeps_schedule_and_training_guard(self):
        root = Path(__file__).resolve().parents[2]
        template = root / "scripts" / "launchd" / "com.dsa.quant-daily-update.plist.template"
        rendered = template.read_text(encoding="utf-8")
        for key, value in {
            "__HOME__": "/tmp/dsa-home",
            "__PYTHON__": "/tmp/dsa/.venv/bin/python",
            "__SCRIPT__": "/tmp/dsa/quant_workflow/scripts/run_daily_update.py",
            "__WORKDIR__": "/tmp/dsa/quant_workflow",
            "__STDOUT__": "/tmp/dsa/quant_workflow/logs/stdout.log",
            "__STDERR__": "/tmp/dsa/quant_workflow/logs/stderr.log",
        }.items():
            rendered = rendered.replace(key, value)

        payload = plistlib.loads(rendered.encode("utf-8"))

        self.assertEqual(payload["Label"], "com.dsa.quant-daily-update")
        arguments = payload["ProgramArguments"]
        self.assertEqual(arguments[:2], ["/usr/bin/env", "-i"])
        self.assertIn("HOME=/tmp/dsa-home", arguments)
        self.assertIn("TZ=Asia/Shanghai", arguments)
        self.assertIn("--skip-if-robust-training", arguments)
        self.assertNotIn("EnvironmentVariables", payload)
        self.assertEqual(
            {(item["Weekday"], item["Hour"], item["Minute"]) for item in payload["StartCalendarInterval"]},
            {(weekday, 18, 40) for weekday in range(1, 6)},
        )
        self.assertFalse(payload["RunAtLoad"])
        self.assertFalse(payload["KeepAlive"])


if __name__ == "__main__":
    unittest.main()
