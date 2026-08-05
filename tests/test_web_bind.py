# -*- coding: utf-8 -*-
"""Tests for Web UI bind host publication helpers."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from src.web_bind import declared_bind_host, infer_uvicorn_cli_host, publish_bind_host


class WebBindTestCase(unittest.TestCase):
    def test_infer_uvicorn_cli_host_from_split_arg(self) -> None:
        self.assertEqual(
            infer_uvicorn_cli_host(["uvicorn", "api.app:app", "--host", "0.0.0.0"]),
            "0.0.0.0",
        )

    def test_infer_uvicorn_cli_host_from_equals_arg(self) -> None:
        self.assertEqual(
            infer_uvicorn_cli_host(["uvicorn", "server:app", "--host=127.0.0.1"]),
            "127.0.0.1",
        )

    def test_infer_uvicorn_cli_host_ignores_args_after_separator(self) -> None:
        self.assertEqual(
            infer_uvicorn_cli_host(["uvicorn", "server:app", "--", "--host", "0.0.0.0"]),
            "",
        )

    def test_publish_bind_host_sets_dsa_bind_host(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            publish_bind_host("127.0.0.1")

            self.assertEqual(os.environ["DSA_BIND_HOST"], "127.0.0.1")
            self.assertEqual(declared_bind_host(), "127.0.0.1")

    def test_webui_main_publishes_bind_host_before_uvicorn_run(self) -> None:
        import webui

        with patch.dict(
            os.environ,
            {
                "WEBUI_HOST": "0.0.0.0",
                "WEBUI_PORT": "8011",
            },
            clear=False,
        ), patch("uvicorn.run") as uvicorn_run:
            os.environ.pop("DSA_BIND_HOST", None)

            exit_code = webui.main()
            self.assertEqual(os.environ["DSA_BIND_HOST"], "0.0.0.0")

        self.assertEqual(exit_code, 0)
        uvicorn_run.assert_called_once()
        self.assertEqual(uvicorn_run.call_args.kwargs["host"], "0.0.0.0")

    def test_server_import_publishes_uvicorn_cli_host_before_app_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("ADMIN_AUTH_ENABLED=false\n", encoding="utf-8")
            code = textwrap.dedent(
                """
                import os
                import sys

                sys.argv = ["uvicorn", "server:app", "--host", "0.0.0.0"]
                os.environ.pop("DSA_BIND_HOST", None)
                os.environ.pop("UVICORN_HOST", None)
                try:
                    import server  # noqa: F401
                except RuntimeError as exc:
                    print(str(exc))
                    print("DSA_BIND_HOST=" + str(os.environ.get("DSA_BIND_HOST")))
                    raise
                """
            )
            env = {
                **os.environ,
                "CORS_ALLOW_ALL": "true",
                "ENV_FILE": str(env_path),
                "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            }
            env.pop("DSA_BIND_HOST", None)
            env.pop("UVICORN_HOST", None)

            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("bound to 0.0.0.0", result.stdout + result.stderr)
        self.assertIn("DSA_BIND_HOST=0.0.0.0", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
