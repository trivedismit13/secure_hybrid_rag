"""
Cross-Platform Multi-Node Orchestrator for SHIELD-RAG.

Launches the 4 isolated microservice nodes:
- KDC Node: Port 8001
- Revocation Proxy: Port 8002
- Cloud Storage Oracle: Port 8003
- Query Gateway: Port 8000

Usage:
    python launch_nodes.py            # Opens 4 dedicated terminal windows (Windows)
    python launch_nodes.py --process   # Runs as multiplexed background subprocesses
"""

import argparse
import os
import subprocess
import sys
import time

NODES = [
    {"name": "KDC Authority", "module": "services.kdc_node", "port": 8001, "color": "0B"},
    {"name": "Revocation Proxy", "module": "services.proxy_node", "port": 8002, "color": "0E"},
    {"name": "Cloud Storage Oracle", "module": "services.storage_node", "port": 8003, "color": "09"},
    {"name": "Query Gateway", "module": "services.gateway_node", "port": 8000, "color": "0A"},
]


def launch_in_windows() -> None:
    """Launches each microservice node in its own visible Windows command prompt."""
    print("=" * 70)
    print(" Spawning 4 Dedicated Microservice Terminals...")
    print("=" * 70)

    for node in NODES:
        title = f"SHIELD-RAG [{node['name']}] :{node['port']}"
        cmd = f'title {title} && color {node["color"]} && python -m {node["module"]}'
        print(f" -> Launching {node['name']} on port {node['port']}...")
        subprocess.Popen(
            ["cmd.exe", "/c", "start", title, "cmd.exe", "/k", cmd],
            shell=True,
        )
        time.sleep(0.8)

    print("\n" + "=" * 70)
    print(" [READY] All 4 Microservices Active:")
    print("  * KDC Authority:       http://localhost:8001/health")
    print("  * Revocation Proxy:    http://localhost:8002/health")
    print("  * Storage Oracle:      http://localhost:8003/health")
    print("  * Query Gateway Docs:  http://localhost:8000/docs")
    print("  * Run Live Demo:       python demo.py")
    print("=" * 70)


def launch_supervised() -> None:
    """Runs all 4 nodes as managed subprocesses in a single terminal."""
    print("=" * 70)
    print(" Running 4 Nodes in Supervised Subprocess Mode (Press Ctrl+C to stop)...")
    print("=" * 70)

    procs = []
    try:
        for node in NODES:
            print(f" -> Starting {node['name']} on :{node['port']}...")
            p = subprocess.Popen(
                [sys.executable, "-m", node["module"]],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            procs.append((node, p))
            time.sleep(0.5)

        print("\nAll 4 nodes running. Streaming logs...\n")
        while True:
            for node, p in procs:
                if p.poll() is not None:
                    print(f"[!] Warning: {node['name']} exited with code {p.returncode}")
                # Non-blocking read line
                line = p.stdout.readline()
                if line:
                    print(f"[{node['name'][:10]}] {line.strip()}")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nShutting down microservice nodes...")
        for _, p in procs:
            p.terminate()
            p.wait()
        print("All microservices cleanly terminated.")


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="SHIELD-RAG Microservice Launcher")
    parser.add_argument(
        "--process",
        action="store_true",
        help="Run as supervised child processes instead of separate terminal windows",
    )
    args = parser.parse_args()

    if args.process or sys.platform != "win32":
        launch_supervised()
    else:
        launch_in_windows()


if __name__ == "__main__":
    main()
