#!/bin/bash
# Monitor Solution A progress, then run B and C sequentially
# Reports every 30 minutes

PROJECT_DIR="/Users/guning/code/competitions/ccl2026-dimt"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
RUN_SCRIPT="$PROJECT_DIR/src/run_all_solutions.py"
OUTPUT_DIR="$PROJECT_DIR/outputs"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# SSL fix
export SSL_CERT_FILE=$($VENV_PYTHON -c "import certifi; print(certifi.where())")

report_progress() {
    local solution=$1
    local lang_dir="$OUTPUT_DIR/results_${solution}/en"
    if [ -d "$lang_dir" ]; then
        local count=$(ls "$lang_dir" 2>/dev/null | wc -l | tr -d ' ')
        local total=500
        local pct=$((count * 100 / total))
        echo "[$(date '+%H:%M:%S')] $solution: $count/$total ($pct%)"
    else
        echo "[$(date '+%H:%M:%S')] $solution: not started yet"
    fi
}

# Phase 1: Monitor Solution A (already running as PID 13675)
echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monitoring Solution A (PID 13675)"
echo "=========================================="

SOL_A_PID=13675
LAST_REPORT=$(date +%s)

while kill -0 $SOL_A_PID 2>/dev/null; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - LAST_REPORT))
    if [ $ELAPSED -ge 1800 ]; then
        echo ""
        echo "--- Progress Report $(date '+%H:%M:%S') ---"
        report_progress solution_a
        # Estimate time remaining
        COUNT=$(ls "$OUTPUT_DIR/results_solution_a/en/" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$COUNT" -gt 0 ]; then
            REMAINING=$((500 - COUNT))
            # Get process elapsed time in seconds
            ELAPSED_SEC=$(ps -p $SOL_A_PID -o etime= 2>/dev/null | tr -d ' ')
            echo "  Process elapsed: $ELAPSED_SEC, remaining images: $REMAINING"
        fi
        LAST_REPORT=$NOW
    fi
    sleep 60
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Solution A process has finished!"
report_progress solution_a
echo ""

# Phase 2: Run Solution B
echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Solution B"
echo "=========================================="

$VENV_PYTHON $RUN_SCRIPT --solution solution_b --skip_existing 2>&1 | tee "$LOG_DIR/solution_b.log" &
SOL_B_PID=$!
echo "Solution B PID: $SOL_B_PID"

LAST_REPORT=$(date +%s)
while kill -0 $SOL_B_PID 2>/dev/null; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - LAST_REPORT))
    if [ $ELAPSED -ge 1800 ]; then
        echo ""
        echo "--- Progress Report $(date '+%H:%M:%S') ---"
        report_progress solution_a
        report_progress solution_b
        LAST_REPORT=$NOW
    fi
    sleep 60
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Solution B has finished!"
report_progress solution_b
echo ""

# Phase 3: Run Solution C
echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Solution C"
echo "=========================================="

$VENV_PYTHON $RUN_SCRIPT --solution solution_c --skip_existing 2>&1 | tee "$LOG_DIR/solution_c.log" &
SOL_C_PID=$!
echo "Solution C PID: $SOL_C_PID"

LAST_REPORT=$(date +%s)
while kill -0 $SOL_C_PID 2>/dev/null; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - LAST_REPORT))
    if [ $ELAPSED -ge 1800 ]; then
        echo ""
        echo "--- Progress Report $(date '+%H:%M:%S') ---"
        report_progress solution_a
        report_progress solution_b
        report_progress solution_c
        LAST_REPORT=$NOW
    fi
    sleep 60
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Solution C has finished!"
report_progress solution_c
echo ""
echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL SOLUTIONS COMPLETE!"
echo "=========================================="
report_progress solution_a
report_progress solution_b
report_progress solution_c
