#!/usr/bin/env python3
"""
DATect System Launcher — starts backend, frontend, opens browser.

Toolchain (best available on PATH / in env, one pass each):
  Python deps : uv pip install -r requirements.txt  →  else pip
  Frontend    : bun install / bun run dev             →  else npm
  ASGI server : granian                               →  else uvicorn

Install https://github.com/astral-sh/uv, https://bun.sh, and use requirements.txt
(includes granian) for the fastest path; npm/pip still work if those are missing.
"""

from __future__ import annotations

import os
import signal
import shutil
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass(frozen=True)
class Toolchain:
    python_installer: str  # "uv" | "pip"
    frontend_pm: str  # "bun" | "npm"
    asgi: str  # "granian" | "uvicorn"


def _which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _granian_importable() -> bool:
    try:
        subprocess.run(
            [sys.executable, "-c", "import granian"],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def resolve_toolchain() -> Toolchain:
    """Pick the fastest option that exists; no user-facing toggles."""
    py = "uv" if _which("uv") else "pip"
    fe = "bun" if _which("bun") else "npm"
    asgi = "granian" if _granian_importable() else "uvicorn"
    return Toolchain(py, fe, asgi)


class DATectLauncher:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.project_root = Path(__file__).resolve().parent
        self.toolchain = resolve_toolchain()

    def _print_toolchain(self) -> None:
        print(
            f"Toolchain: {self.toolchain.python_installer} (Python) + "
            f"{self.toolchain.frontend_pm} (frontend) + {self.toolchain.asgi} (API)"
        )

    def check_port(self, port: int) -> None:
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if not result.stdout.strip():
                return
            pids = result.stdout.strip().split("\n")
            print(f"Port {port} is busy. Stopping existing processes...")
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    time.sleep(1)
                except (ProcessLookupError, ValueError, PermissionError):
                    pass
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

    def wait_for_service(self, url: str, name: str, max_wait: int = 30) -> bool:
        print(f"Waiting for {name}...")
        for _ in range(max_wait):
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    print(f"{name} ready")
                    return True
            except OSError:
                pass
            time.sleep(1)
        print(f"{name} failed to start within {max_wait} seconds")
        return False

    def check_prerequisites(self) -> bool:
        print("Checking prerequisites...")
        data_file = self.project_root / "data/processed/final_output.parquet"
        if not data_file.exists():
            print("Data file not found at data/processed/final_output.parquet")
            print("Please run 'python dataset-creation.py' first")
            return False
        try:
            subprocess.run(["node", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Node.js is not installed (required for Bun/npm)")
            print("Install from https://nodejs.org or: brew install node")
            return False
        return True

    def install_dependencies(self) -> bool:
        print("Installing dependencies...")
        req = self.project_root / "requirements.txt"
        if not req.is_file():
            print(f"Missing {req.name} at project root")
            return False

        try:
            if self.toolchain.python_installer == "uv":
                subprocess.run(
                    ["uv", "pip", "install", "--quiet", "-r", str(req)],
                    check=True,
                    capture_output=True,
                )
            else:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--quiet",
                        "-r",
                        str(req),
                    ],
                    check=True,
                    capture_output=True,
                )
        except subprocess.CalledProcessError as e:
            print(f"Python dependency install failed: {e}")
            return False

        frontend_dir = self.project_root / "frontend"
        if not (frontend_dir / "node_modules").is_dir():
            try:
                if self.toolchain.frontend_pm == "bun":
                    subprocess.run(
                        ["bun", "install"],
                        cwd=frontend_dir,
                        check=True,
                        capture_output=True,
                    )
                else:
                    subprocess.run(
                        ["npm", "install"],
                        cwd=frontend_dir,
                        check=True,
                        capture_output=True,
                    )
            except subprocess.CalledProcessError:
                print("Failed to install frontend dependencies")
                return False

        # Refresh ASGI choice now that granian may have been installed
        asgi = "granian" if _granian_importable() else "uvicorn"
        self.toolchain = Toolchain(
            self.toolchain.python_installer,
            self.toolchain.frontend_pm,
            asgi,
        )
        return True

    def start_backend(self) -> bool:
        env = os.environ.copy()
        env.setdefault("CACHE_DIR", str(self.project_root / "cache"))

        if self.toolchain.asgi == "granian":
            print("Starting backend with Granian...")
            self.backend_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "granian",
                    "--interface",
                    "asgi",
                    "backend.api:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8000",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.project_root,
                env=env,
            )
        else:
            print("Starting backend with Uvicorn...")
            self.backend_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "backend.api:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8000",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.project_root,
                env=env,
            )
        return self.wait_for_service("http://localhost:8000/health", "Backend")

    def start_frontend(self) -> bool:
        frontend_dir = self.project_root / "frontend"
        if self.toolchain.frontend_pm == "bun":
            print("Starting frontend with Bun...")
            self.frontend_process = subprocess.Popen(
                ["bun", "run", "dev"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=frontend_dir,
            )
        else:
            print("Starting frontend with npm...")
            self.frontend_process = subprocess.Popen(
                ["npm", "run", "dev"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=frontend_dir,
            )
        return self.wait_for_service("http://localhost:3000", "Frontend", max_wait=45)

    def open_browser(self) -> None:
        print("Opening browser...")
        time.sleep(1)
        try:
            webbrowser.open("http://localhost:3000")
        except OSError:
            print("Could not open browser. Visit: http://localhost:3000")

    def cleanup(self) -> None:
        print("Shutting down...")
        if self.backend_process:
            self.backend_process.terminate()
            time.sleep(1)
            if self.backend_process.poll() is None:
                self.backend_process.kill()
        if self.frontend_process:
            self.frontend_process.terminate()
            time.sleep(1)
            if self.frontend_process.poll() is None:
                self.frontend_process.kill()
        for port in (8000, 3000):
            try:
                result = subprocess.run(
                    ["lsof", "-ti", f":{port}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if not result.stdout.strip():
                    continue
                for pid in result.stdout.strip().split("\n"):
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except (ProcessLookupError, ValueError, PermissionError):
                        pass
            except (FileNotFoundError, subprocess.SubprocessError):
                pass

    def run(self) -> bool:
        try:
            print("DATect System Launcher")
            print("=============================")
            self.check_port(8000)
            self.check_port(3000)
            if not self.check_prerequisites():
                return False
            if not self.install_dependencies():
                return False
            self._print_toolchain()
            if not self.start_backend():
                print("Backend failed to start")
                return False
            if not self.start_frontend():
                print("Frontend failed to start")
                return False
            self.open_browser()
            print("\nDATect is now running!")
            print("Frontend: http://localhost:3000")
            print("Backend: http://localhost:8000")
            print("API Docs: http://localhost:8000/docs")
            print("\nPress Ctrl+C to stop")
            try:
                while True:
                    if self.backend_process.poll() is not None:
                        print("Backend stopped unexpectedly")
                        break
                    if self.frontend_process.poll() is not None:
                        print("Frontend stopped unexpectedly")
                        break
                    time.sleep(2)
            except KeyboardInterrupt:
                print("\nShutting down...")
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False
        finally:
            self.cleanup()


def main() -> None:
    """Entry point for `python -m run_datect` and the `datect` console script."""
    success = DATectLauncher().run()
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
