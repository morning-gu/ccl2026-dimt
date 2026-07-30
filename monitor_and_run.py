#!/usr/bin/env python3
"""Monitor Solution A, then run B and C sequentially with 30-min progress reports."""
import os, sys, time, subprocess, glob
from pathlib import Path
from datetime import datetime

PROJECT = Path("/Users/guning/code/competitions/ccl2026-dimt")
VENV_PY = str(PROJECT / ".venv" / "bin" / "python")
RUN_SCRIPT = str(PROJECT / "src" / "run_all_solutions.py")
OUTPUTS = PROJECT / "outputs"
LOGS = PROJECT / "logs"
LOGS.mkdir(exist_ok=True)

# SSL fix
os.environ["SSL_CERT_FILE"] = subprocess.check_output(
    [VENV_PY, "-c", "import certifi; print(certifi.where())"],
    text=True
).strip()

def count_outputs(solution):
    d = OUTPUTS / f"results_{solution}" / "en"
    if d.is_dir():
        return len(list(d.glob("*.jpg")))
    return 0

def report(solutions):
    now = datetime.now().strftime("%H:%M:%S")
    lines = [f"[{now}] Progress:"]
    for s in solutions:
        c = count_outputs(s)
        pct = c * 100 // 500
        lines.append(f"  {s}: {c}/500 ({pct}%)")
    msg = "\n".join(lines)
    print(msg, flush=True)
    return msg

def wait_for_pid(pid, solution, interval=1800):
    """Wait for a PID to finish, reporting every `interval` seconds."""
    last_report = time.time()
    while True:
        try:
            os.kill(pid, 0)  # Check if process exists
        except ProcessLookupError:
            return
        now = time.time()
        if now - last_report >= interval:
            report([solution])
            last_report = now
        time.sleep(30)

def run_solution(solution):
    log_path = LOGS / f"{solution}.log"
    print(f"\n{'='*50}", flush=True)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Starting {solution}", flush=True)
    print(f"{'='*50}", flush=True)
    
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            [VENV_PY, RUN_SCRIPT, "--solution", solution, "--skip_existing"],
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    print(f"PID: {proc.pid}", flush=True)
    
    last_report = time.time()
    while proc.poll() is None:
        now = time.time()
        if now - last_report >= 1800:
            report([solution])
            last_report = now
        time.sleep(30)
    
    print(f"[{datetime.now():%H:%M:%S}] {solution} finished (exit code {proc.returncode})", flush=True)
    report([solution])

# --- Main ---
print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Monitor started", flush=True)

# Phase 1: Monitor Solution A (PID 13675)
SOL_A_PID = 13675
print(f"Monitoring Solution A (PID {SOL_A_PID})...", flush=True)
report(["solution_a"])
wait_for_pid(SOL_A_PID, "solution_a")
print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] Solution A complete!", flush=True)
report(["solution_a"])

# Phase 2: Run Solution B
run_solution("solution_b")

# Phase 3: Run Solution C
run_solution("solution_c")

# Done
print(f"\n{'='*50}", flush=True)
print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] ALL SOLUTIONS COMPLETE!", flush=True)
print(f"{'='*50}", flush=True)
report(["solution_a", "solution_b", "solution_c"])
