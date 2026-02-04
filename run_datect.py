#!/usr/bin/env python3
"""
DATect System Launcher
Starts backend, frontend, and opens browser

Optimizations:
- UV package manager support (10x faster installs)
- Granian ASGI server (20-40% faster than Uvicorn)
- Bun support for frontend (faster than npm)
"""

import subprocess
import time
import os
import sys
import signal
import webbrowser
import shutil
import requests
from pathlib import Path


def check_command_exists(cmd: str) -> bool:
    """Check if a command is available in PATH."""
    return shutil.which(cmd) is not None


class DATectLauncher:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.project_root = Path(__file__).parent
        
        # Auto-detect best tools available
        self.use_uv = check_command_exists('uv')
        self.use_bun = check_command_exists('bun')
        self.use_granian = self._check_granian_available()
        
        if self.use_uv:
            print("UV package manager detected (10x faster installs)")
        if self.use_bun:
            print("Bun detected (faster frontend builds)")
        if self.use_granian:
            print("Granian server available (20-40% faster API)")
    
    def _check_granian_available(self) -> bool:
        """Check if Granian is installed."""
        try:
            subprocess.run(
                [sys.executable, '-c', 'import granian'],
                check=True, capture_output=True
            )
            return True
        except:
            return False
    
    def check_port(self, port):
        """Kill existing processes on port if busy"""
        try:
            result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                print(f"Port {port} is busy. Stopping existing processes...")
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        time.sleep(1)
                    except:
                        pass
        except:
            pass
    
    def wait_for_service(self, url, name, max_wait=30):
        """Wait for service to respond with HTTP 200"""
        print(f"Waiting for {name}...")
        for _ in range(max_wait):
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    print(f"{name} ready")
                    return True
            except:
                pass
            time.sleep(1)
        
        print(f"{name} failed to start within {max_wait} seconds")
        return False
    
    def check_prerequisites(self):
        """Validate basic requirements"""
        print("Checking prerequisites...")
        
        data_file = self.project_root / "data/processed/final_output.parquet"
        if not data_file.exists():
            print("Data file not found at data/processed/final_output.parquet")
            print("Please run 'python dataset-creation.py' first")
            return False
        
        try:
            subprocess.run(['node', '--version'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Node.js is not installed")
            print("Install from https://nodejs.org or run: brew install node")
            return False
            
        return True
    
    
    def install_dependencies(self):
        """Install required packages using fastest available tools."""
        print("Installing dependencies...")
        
        # Install Python dependencies
        try:
            if self.use_uv:
                # UV is ~10x faster than pip
                print("Using UV for fast Python package installation...")
                subprocess.run([
                    'uv', 'pip', 'install', '--quiet', 
                    'fastapi', 'uvicorn', 'granian', 'pandas', 'scikit-learn', 
                    'plotly', 'xgboost', 'polars'
                ], check=True, capture_output=True)
            else:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', '--quiet', 
                    'fastapi', 'uvicorn', 'pandas', 'scikit-learn', 'plotly', 'xgboost'
                ], check=True, capture_output=True)
        except Exception as e:
            print(f"Warning: Could not install some Python packages: {e}")
        
        # Install frontend dependencies
        frontend_dir = self.project_root / "frontend"
        if not (frontend_dir / "node_modules").exists():
            try:
                if self.use_bun:
                    # Bun is much faster than npm
                    print("Using Bun for fast frontend installation...")
                    subprocess.run(['bun', 'install'], 
                                 cwd=frontend_dir, check=True, capture_output=True)
                else:
                    subprocess.run(['npm', 'install'], 
                                 cwd=frontend_dir, check=True, capture_output=True)
            except subprocess.CalledProcessError:
                print("Failed to install frontend dependencies")
                return False
        
        # Update granian availability after installing
        self.use_granian = self._check_granian_available()
        
        return True
    
    def start_backend(self):
        """Launch FastAPI backend server using fastest available ASGI server."""
        
        if self.use_granian:
            print("Starting backend with Granian (Rust-based, 20-40% faster)...")
            self.backend_process = subprocess.Popen([
                'granian', '--interface', 'asgi', 
                'backend.api:app', 
                '--host', '0.0.0.0', 
                '--port', '8000'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.project_root)
        else:
            print("Starting backend with Uvicorn...")
            self.backend_process = subprocess.Popen([
                sys.executable, '-m', 'uvicorn',
                'backend.api:app',
                '--host', '0.0.0.0',
                '--port', '8000'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.project_root)
        
        return self.wait_for_service("http://localhost:8000/health", "Backend")
    
    def start_frontend(self):
        """Launch React development server using fastest available tool."""
        
        frontend_dir = self.project_root / "frontend"
        
        if self.use_bun:
            print("Starting frontend with Bun (faster builds)...")
            self.frontend_process = subprocess.Popen([
                'bun', 'run', 'dev'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=frontend_dir)
        else:
            print("Starting frontend with npm...")
            self.frontend_process = subprocess.Popen([
                'npm', 'run', 'dev'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=frontend_dir)
        
        return self.wait_for_service("http://localhost:3000", "Frontend", max_wait=45)
    
    def open_browser(self):
        """Launch browser to application URL"""
        print("Opening browser...")
        time.sleep(1)
        try:
            webbrowser.open("http://localhost:3000")
        except:
            print("Could not open browser. Visit: http://localhost:3000")
    
    def cleanup(self):
        """Terminate all spawned processes"""
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
        
        for port in [8000, 3000]:
            try:
                result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                      capture_output=True, text=True)
                if result.stdout.strip():
                    for pid in result.stdout.strip().split('\n'):
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                        except:
                            pass
            except:
                pass
    
    def run(self):
        """Execute complete system startup sequence"""
        try:
            print("DATect System Launcher")
            print("=============================")
            
            self.check_port(8000)
            self.check_port(3000)
            
            if not self.check_prerequisites():
                return False
            
            if not self.install_dependencies():
                return False
            
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

if __name__ == "__main__":
    launcher = DATectLauncher()
    success = launcher.run()
    sys.exit(0 if success else 1)