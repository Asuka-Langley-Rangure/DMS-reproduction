from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts import planner_smoke


def make_args(**overrides) -> argparse.Namespace:
    defaults = {
        "task": "ContactsAddContact",
        "output_dir": "planner_smoke_runs",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "dms-qwen-secret",
        "model": "qwen2.5-vl-7b",
        "max_tokens": 512,
        "timeout": 120,
        "temperature": 0.0,
        "ssh_host": "114.212.165.149",
        "ssh_user": "chencen",
        "ssh_password": "",
        "local_port": 8000,
        "remote_port": 8007,
        "skip_ssh_tunnel": False,
        "healthcheck_timeout": 5,
        "ssh_ready_timeout": 20,
        "console_port": 5554,
        "grpc_port": 8554,
        "adb_path": r"D:\Android\Sdk\platform-tools\adb.exe",
        "perform_emulator_setup": False,
        "emulator_start_script": r"F:\baoyantest\dms\start_androidworld_emulator.bat",
        "skip_emulator_launch": False,
        "emulator_ready_timeout": 120,
        "a11y_apk": r"F:\baoyantest\dms\android_world\third_party\a11y_forwarder.apk",
        "max_ui_elements": 50,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class PlannerSmokeConfigTest(unittest.TestCase):
    def test_healthcheck_url(self) -> None:
        self.assertEqual(
            planner_smoke.healthcheck_url("http://127.0.0.1:8000/v1"),
            "http://127.0.0.1:8000/v1/models",
        )

    def test_build_ssh_command(self) -> None:
        args = make_args()
        self.assertEqual(
            planner_smoke.build_ssh_command(args),
            [
                "ssh",
                "-N",
                "-L",
                "8000:127.0.0.1:8007",
                "chencen@114.212.165.149",
            ],
        )

    def test_make_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = planner_smoke.make_run_dir(tmpdir, "ContactsAddContact")
            self.assertTrue(run_dir.exists())
            self.assertEqual(run_dir.parent, Path(tmpdir))
            self.assertIn("ContactsAddContact_", run_dir.name)

    @patch("scripts.planner_smoke.healthcheck_model_endpoint", return_value=(True, "ok"))
    def test_reuse_existing_model_endpoint_without_starting_ssh(self, healthcheck: MagicMock) -> None:
        args = make_args()
        meta = {"tunnel_reused": False, "tunnel_started_by_script": False}

        result = planner_smoke.ensure_ssh_tunnel(args, meta)

        self.assertIsNone(result)
        self.assertTrue(meta["tunnel_reused"])
        self.assertFalse(meta["tunnel_started_by_script"])
        healthcheck.assert_called_once()

    @patch("scripts.planner_smoke.time.sleep", return_value=None)
    @patch("scripts.planner_smoke.subprocess.Popen")
    @patch("scripts.planner_smoke.shutil.which", return_value=r"C:\Windows\System32\OpenSSH\ssh.exe")
    @patch(
        "scripts.planner_smoke.healthcheck_model_endpoint",
        side_effect=[(False, "connection_error"), (True, "ok")],
    )
    def test_start_ssh_when_model_endpoint_unreachable(
        self,
        healthcheck: MagicMock,
        which: MagicMock,
        popen: MagicMock,
        sleep: MagicMock,
    ) -> None:
        process = MagicMock()
        process.poll.return_value = None
        popen.return_value = process

        args = make_args()
        meta = {"tunnel_reused": False, "tunnel_started_by_script": False}

        result = planner_smoke.ensure_ssh_tunnel(args, meta)

        self.assertIs(result, process)
        self.assertTrue(meta["tunnel_started_by_script"])
        popen.assert_called_once()
        which.assert_called_once_with("ssh")
        command = popen.call_args.args[0]
        self.assertEqual(
            command,
            ["ssh", "-N", "-L", "8000:127.0.0.1:8007", "chencen@114.212.165.149"],
        )
        self.assertGreaterEqual(healthcheck.call_count, 2)

    def test_build_meta_contains_paths_and_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = make_args()
            meta = planner_smoke.build_meta(Path(tmpdir), args)
            self.assertEqual(meta["task"], "ContactsAddContact")
            self.assertEqual(meta["model"], "qwen2.5-vl-7b")
            self.assertEqual(meta["base_url"], "http://127.0.0.1:8000/v1")
            self.assertIn("planner_smoke_runs", args.output_dir)
            self.assertEqual(
                meta["ssh_command"],
                ["ssh", "-N", "-L", "8000:127.0.0.1:8007", "chencen@114.212.165.149"],
            )
            self.assertFalse(meta["ssh_password_present"])
            self.assertFalse(meta["emulator_launched_by_script"])

    @patch("scripts.planner_smoke.adb_device_ready", return_value=True)
    @patch("scripts.planner_smoke.is_tcp_port_open", return_value=True)
    @patch("scripts.planner_smoke.find_adb_path", return_value=r"D:\Android\Sdk\platform-tools\adb.exe")
    def test_reuse_existing_emulator_when_ready(
        self,
        find_adb_path: MagicMock,
        is_tcp_port_open: MagicMock,
        adb_device_ready: MagicMock,
    ) -> None:
        args = make_args()
        meta = {"emulator_launched_by_script": False}

        planner_smoke.ensure_emulator_running(args, meta)

        self.assertFalse(meta["emulator_launched_by_script"])
        find_adb_path.assert_called_once()

    @patch("scripts.planner_smoke.time.sleep", return_value=None)
    @patch("scripts.planner_smoke.subprocess.Popen")
    @patch("scripts.planner_smoke.adb_device_ready", side_effect=[False, False, True])
    @patch("scripts.planner_smoke.is_tcp_port_open", side_effect=[False, False, True])
    @patch("scripts.planner_smoke.find_adb_path", return_value=r"D:\Android\Sdk\platform-tools\adb.exe")
    @patch("pathlib.Path.is_file", return_value=True)
    def test_launch_emulator_script_when_not_ready(
        self,
        path_is_file: MagicMock,
        find_adb_path: MagicMock,
        is_tcp_port_open: MagicMock,
        adb_device_ready: MagicMock,
        popen: MagicMock,
        sleep: MagicMock,
    ) -> None:
        args = make_args()
        meta = {"emulator_launched_by_script": False}

        planner_smoke.ensure_emulator_running(args, meta)

        self.assertTrue(meta["emulator_launched_by_script"])
        popen.assert_called_once()
        command = popen.call_args.args[0]
        self.assertEqual(command[:3], ["cmd", "/c", "start"])

    @patch("scripts.planner_smoke.time.sleep", return_value=None)
    @patch("scripts.planner_smoke.build_paramiko_tunnel")
    @patch(
        "scripts.planner_smoke.healthcheck_model_endpoint",
        side_effect=[(False, "connection_error"), (True, "ok")],
    )
    def test_use_paramiko_tunnel_when_password_is_provided(
        self,
        healthcheck: MagicMock,
        build_paramiko_tunnel: MagicMock,
        sleep: MagicMock,
    ) -> None:
        class FakeTunnel:
            def terminate(self) -> None:
                return None

        tunnel = FakeTunnel()
        build_paramiko_tunnel.return_value = tunnel

        args = make_args(ssh_password="123456")
        meta = {"tunnel_reused": False, "tunnel_started_by_script": False, "tunnel_mode": None}

        result = planner_smoke.ensure_ssh_tunnel(args, meta)

        self.assertIs(result, tunnel)
        self.assertTrue(meta["tunnel_started_by_script"])
        self.assertEqual(meta["tunnel_mode"], "paramiko_password")
        build_paramiko_tunnel.assert_called_once_with(
            ssh_host="114.212.165.149",
            ssh_user="chencen",
            ssh_password="123456",
            local_port=8000,
            remote_port=8007,
        )


if __name__ == "__main__":
    unittest.main()
