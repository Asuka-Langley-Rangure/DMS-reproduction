from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import select
import shutil
import socket
import socketserver
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
VENDOR_DIR = ROOT_DIR / ".vendor"
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from dms_reproduction.agents.planner import AndroidTaskPlanner


DEFAULT_A11Y_APK = (
    ROOT_DIR / "android_world" / "third_party" / "a11y_forwarder.apk"
)


class ParamikoTunnel:
    """Minimal local port forward wrapper backed by paramiko."""

    def __init__(
        self,
        server: socketserver.ThreadingTCPServer,
        thread: threading.Thread,
        ssh_client: Any,
    ) -> None:
        self.server = server
        self.thread = thread
        self.ssh_client = ssh_client

    def terminate(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)
        self.ssh_client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single real-VLM planner smoke test on AndroidWorld."
    )
    parser.add_argument("--task", default="ContactsAddContact")
    parser.add_argument("--output_dir", default="planner_smoke_runs")
    parser.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key", default="dms-qwen-secret")
    parser.add_argument("--model", default="qwen2.5-vl-7b")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--ssh_host", default="114.212.165.149")
    parser.add_argument("--ssh_user", default="chencen")
    parser.add_argument("--ssh_password", default=os.environ.get("DMS_SSH_PASSWORD", ""))
    parser.add_argument("--local_port", type=int, default=8000)
    parser.add_argument("--remote_port", type=int, default=8007)
    parser.add_argument("--skip_ssh_tunnel", action="store_true")
    parser.add_argument("--healthcheck_timeout", type=int, default=5)
    parser.add_argument("--ssh_ready_timeout", type=int, default=20)

    parser.add_argument("--console_port", type=int, default=5554)
    parser.add_argument("--grpc_port", type=int, default=8554)
    parser.add_argument("--adb_path", default=r"D:\Android\Sdk\platform-tools\adb.exe")
    parser.add_argument("--perform_emulator_setup", action="store_true")
    parser.add_argument(
        "--emulator_start_script",
        default=str(ROOT_DIR / "start_androidworld_emulator.bat"),
    )
    parser.add_argument("--skip_emulator_launch", action="store_true")
    parser.add_argument("--emulator_ready_timeout", type=int, default=120)
    parser.add_argument("--a11y_apk", default=str(DEFAULT_A11Y_APK))
    parser.add_argument("--max_ui_elements", type=int, default=50)
    return parser.parse_args()


def build_ssh_command(args: argparse.Namespace) -> list[str]:
    return [
        "ssh",
        "-N",
        "-L",
        f"{args.local_port}:127.0.0.1:{args.remote_port}",
        f"{args.ssh_user}@{args.ssh_host}",
    ]


def build_meta(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "task": args.task,
        "model": args.model,
        "base_url": args.base_url,
        "api_key_present": bool(args.api_key),
        "ssh_password_present": bool(args.ssh_password),
        "console_port": args.console_port,
        "grpc_port": args.grpc_port,
        "emulator_start_script": args.emulator_start_script,
        "emulator_launched_by_script": False,
        "output_dir": str(run_dir),
        "timestamp": datetime.now().isoformat(),
        "tunnel_reused": False,
        "tunnel_started_by_script": False,
        "planner_call_seconds": None,
        "healthcheck_url": healthcheck_url(args.base_url),
        "ssh_command": build_ssh_command(args),
        "tunnel_mode": None,
        "error": None,
    }


def healthcheck_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/models"


def healthcheck_model_endpoint(
    base_url: str,
    api_key: str,
    timeout: int,
) -> tuple[bool, str]:
    import requests

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(
            healthcheck_url(base_url),
            headers=headers,
            timeout=timeout,
        )
    except requests.Timeout:
        return False, "timeout"
    except requests.ConnectionError:
        return False, "connection_error"
    except requests.RequestException as exc:
        return False, f"request_error:{exc}"

    if response.status_code == 200:
        return True, "ok"
    if response.status_code == 401:
        return False, "unauthorized"
    if response.status_code == 403:
        return False, "forbidden"
    return False, f"http_{response.status_code}"


def build_popen_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return kwargs


def is_tcp_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def build_paramiko_tunnel(
    ssh_host: str,
    ssh_user: str,
    ssh_password: str,
    local_port: int,
    remote_port: int,
) -> ParamikoTunnel:
    try:
        import paramiko
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "paramiko is not installed. Install it into the project or omit --ssh_password."
        ) from exc

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(
        hostname=ssh_host,
        username=ssh_user,
        password=ssh_password,
        allow_agent=False,
        look_for_keys=False,
        timeout=10,
        banner_timeout=10,
        auth_timeout=10,
    )
    transport = ssh_client.get_transport()
    if transport is None or not transport.is_active():
        ssh_client.close()
        raise RuntimeError("Paramiko transport is not active after SSH login.")

    class ForwardHandler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            channel = transport.open_channel(
                "direct-tcpip",
                ("127.0.0.1", remote_port),
                self.request.getpeername(),
            )
            try:
                while True:
                    readable, _, _ = select.select([self.request, channel], [], [])
                    if self.request in readable:
                        data = self.request.recv(1024)
                        if not data:
                            break
                        channel.sendall(data)
                    if channel in readable:
                        data = channel.recv(1024)
                        if not data:
                            break
                        self.request.sendall(data)
            finally:
                channel.close()
                self.request.close()

    class ThreadedTCPServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = ThreadedTCPServer(("127.0.0.1", local_port), ForwardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return ParamikoTunnel(server=server, thread=thread, ssh_client=ssh_client)


def ensure_ssh_tunnel(args: argparse.Namespace, meta: dict[str, Any]) -> Any | None:
    healthy, reason = healthcheck_model_endpoint(
        args.base_url,
        args.api_key,
        args.healthcheck_timeout,
    )
    if healthy:
        meta["tunnel_reused"] = True
        meta["tunnel_mode"] = "reused"
        return None

    if args.skip_ssh_tunnel:
        raise RuntimeError(
            f"Model endpoint is not reachable at {healthcheck_url(args.base_url)}: {reason}"
        )

    process: Any
    if args.ssh_password:
        try:
            process = build_paramiko_tunnel(
                ssh_host=args.ssh_host,
                ssh_user=args.ssh_user,
                ssh_password=args.ssh_password,
                local_port=args.local_port,
                remote_port=args.remote_port,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to launch password-based SSH tunnel via paramiko: "
                f"{exc}"
            ) from exc
        meta["tunnel_mode"] = "paramiko_password"
    else:
        ssh_binary = shutil.which("ssh")
        if not ssh_binary:
            raise RuntimeError("ssh executable not found in PATH.")

        command = build_ssh_command(args)
        try:
            process = subprocess.Popen(command, **build_popen_kwargs())
        except OSError as exc:
            raise RuntimeError(
                f"Failed to launch SSH tunnel {' '.join(command)}: {exc}"
            ) from exc
        meta["tunnel_mode"] = "ssh_subprocess"

    meta["tunnel_started_by_script"] = True
    deadline = time.time() + args.ssh_ready_timeout
    while time.time() < deadline:
        if hasattr(process, "poll") and process.poll() is not None:
            raise RuntimeError(
                "SSH tunnel exited before the model endpoint became ready. "
                "Please ensure SSH auth is non-interactive and the remote port is reachable."
            )
        healthy, reason = healthcheck_model_endpoint(
            args.base_url,
            args.api_key,
            args.healthcheck_timeout,
        )
        if healthy:
            return process
        time.sleep(1.0)

    process.terminate()
    raise RuntimeError(
        "Timed out waiting for model endpoint after starting SSH tunnel. "
        f"Last healthcheck result: {reason}."
    )


def find_adb_path(explicit_adb_path: str | None) -> str:
    if explicit_adb_path and Path(explicit_adb_path).is_file():
        return explicit_adb_path

    candidates: list[str] = []
    which_adb = shutil.which("adb")
    if which_adb:
        candidates.append(which_adb)

    for env_name in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        sdk_root = os.environ.get(env_name)
        if sdk_root:
            candidates.append(str(Path(sdk_root) / "platform-tools" / "adb.exe"))
            candidates.append(str(Path(sdk_root) / "platform-tools" / "adb"))

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(
            str(Path(local_app_data) / "Android" / "Sdk" / "platform-tools" / "adb.exe")
        )

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("adb not found. Please pass --adb_path explicitly.")


def adb_device_ready(adb_path: str, console_port: int) -> bool:
    device_name = f"emulator-{console_port}"
    try:
        result = subprocess.run(
            [adb_path, "devices"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError:
        return False

    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == device_name and parts[1] == "device":
            return True
    return False


def ensure_emulator_running(args: argparse.Namespace, meta: dict[str, Any]) -> None:
    adb_path = find_adb_path(args.adb_path)
    grpc_ready = is_tcp_port_open("127.0.0.1", args.grpc_port)
    adb_ready = adb_device_ready(adb_path, args.console_port)
    if grpc_ready and adb_ready:
        return

    if args.skip_emulator_launch:
        raise RuntimeError(
            "Emulator is not ready and --skip_emulator_launch was provided. "
            f"Expected adb device emulator-{args.console_port} and grpc port {args.grpc_port}."
        )

    script_path = Path(args.emulator_start_script).resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"Emulator start script not found: {script_path}")

    try:
        subprocess.Popen(
            ["cmd", "/c", "start", "", str(script_path)],
            **build_popen_kwargs(),
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to launch emulator start script {script_path}: {exc}") from exc

    meta["emulator_launched_by_script"] = True
    deadline = time.time() + args.emulator_ready_timeout
    while time.time() < deadline:
        grpc_ready = is_tcp_port_open("127.0.0.1", args.grpc_port)
        adb_ready = adb_device_ready(adb_path, args.console_port)
        if grpc_ready and adb_ready:
            return
        time.sleep(2.0)

    raise RuntimeError(
        "Emulator did not become ready in time after launching the batch script. "
        f"Expected adb device emulator-{args.console_port} and grpc port {args.grpc_port}."
    )


def patch_a11y_forwarder_apk(local_apk: str | None) -> None:
    if not local_apk:
        return

    apk_path = Path(local_apk).resolve()
    if not apk_path.is_file():
        raise FileNotFoundError(f"A11y forwarder APK not found: {apk_path}")

    apk_bytes = apk_path.read_bytes()
    if len(apk_bytes) < 1024 * 1024:
        raise ValueError(
            f"A11y forwarder APK seems too small: {len(apk_bytes)} bytes. "
            "Please verify the file is fully available."
        )

    from android_env.wrappers import a11y_grpc_wrapper

    def _local_get_accessibility_forwarder_apk() -> bytes:
        return apk_bytes

    a11y_grpc_wrapper._get_accessibility_forwarder_apk = _local_get_accessibility_forwarder_apk


def load_env(args: argparse.Namespace) -> Any:
    from android_world.env import env_launcher

    adb_path = find_adb_path(args.adb_path)
    kwargs = {
        "console_port": args.console_port,
        "emulator_setup": args.perform_emulator_setup,
        "adb_path": adb_path,
    }

    try:
        return env_launcher.load_and_setup_env(**kwargs, grpc_port=args.grpc_port)
    except TypeError:
        return env_launcher.load_and_setup_env(**kwargs)


def load_task(task_name: str) -> tuple[Any, str]:
    from android_world import registry

    task_registry = registry.TaskRegistry()
    aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)
    if task_name not in aw_registry:
        raise KeyError(f"Task {task_name!r} not found in AndroidWorld registry.")

    task_type = aw_registry[task_name]
    params = task_type.generate_random_params()
    task = task_type(params)
    return task, str(task.goal)


def make_run_dir(output_dir: str, task: str) -> Path:
    run_name = f"{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def maybe_teardown(task: Any, env: Any) -> None:
    for name in ("tear_down", "teardown"):
        teardown_fn = getattr(task, name, None)
        if callable(teardown_fn):
            teardown_fn(env)
            return


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = make_run_dir(args.output_dir, args.task)
    meta = build_meta(run_dir, args)
    tunnel_process: subprocess.Popen | None = None
    env = None
    task = None

    try:
        from dms_reproduction.llm.base_client import OpenAICompatibleConfig
        from dms_reproduction.llm.openai_compatible import OpenAICompatibleClient
        from dms_reproduction.envs.android_world_adapter import AndroidWorldObservationAdapter

        tunnel_process = ensure_ssh_tunnel(args, meta)
        ensure_emulator_running(args, meta)

        patch_a11y_forwarder_apk(args.a11y_apk)
        env = load_env(args)
        env.reset(go_home=True)

        task, goal = load_task(args.task)
        task.initialize_task(env)

        adapter = AndroidWorldObservationAdapter(max_ui_elements=args.max_ui_elements)
        observation = adapter.capture_observation(
            env,
            goal=goal,
            step_id=0,
            include_screenshots=True,
        )

        llm_client = OpenAICompatibleClient(
            OpenAICompatibleConfig(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
        )
        planner = AndroidTaskPlanner(llm_client)
        planner_messages = planner.build_messages(
            user_goal=goal,
            observation=observation,
            task_history=[],
            memory_context="",
        )
        planner_prompt = planner.extract_user_text_prompt(planner_messages)

        start_time = time.time()
        planner_result = planner.plan(
            user_goal=goal,
            observation=observation,
            task_history=[],
            memory_context="",
        )
        meta["planner_call_seconds"] = round(time.time() - start_time, 3)

        save_json(run_dir / "observation.json", observation)
        save_json(run_dir / "planner_messages.json", planner.messages_to_jsonable(planner_messages))
        save_text(run_dir / "planner_prompt.txt", planner_prompt)
        save_text(run_dir / "planner_raw_response.txt", planner_result.raw_response)
        save_json(run_dir / "planner_result.json", planner_result.to_dict())
        save_json(run_dir / "meta.json", meta)

        print(f"Task: {args.task}")
        print(f"Goal: {goal}")
        print(f"Current activity: {observation.get('current_activity')}")
        print(f"Planner call seconds: {meta['planner_call_seconds']}")
        print(f"Run dir: {run_dir}")
        raw_preview = planner_result.raw_response[:300].replace("\n", " ")
        print(f"Raw response preview: {raw_preview}")
        if planner_result.parse_error:
            print(f"Parse error: {planner_result.parse_error}")
        elif planner_result.is_goal_complete:
            print(f"Goal complete: {planner_result.completion_message}")
        else:
            print("Planner subtasks:")
            for idx, subtask in enumerate(planner_result.subtasks, start=1):
                print(f"{idx}. {subtask.task} | reason={subtask.reason}")

        return {
            "run_dir": str(run_dir),
            "meta": meta,
            "planner_result": planner_result.to_dict(),
        }
    except Exception as exc:
        meta["error"] = str(exc)
        save_json(run_dir / "meta.json", meta)
        raise
    finally:
        if task is not None and env is not None:
            try:
                maybe_teardown(task, env)
            except Exception:
                pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if tunnel_process is not None and meta.get("tunnel_started_by_script"):
            try:
                tunnel_process.terminate()
            except Exception:
                pass


def main() -> int:
    args = parse_args()
    try:
        run_smoke(args)
    except Exception as exc:
        print(f"[planner_smoke] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
