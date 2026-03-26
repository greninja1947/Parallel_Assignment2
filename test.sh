#!/bin/bash
# ============================================================
# HPC Test Runner for Assignment 2
# Usage: ./test <knn|kmeans|approx_knn|all>
#        ./test correctness   (run full correctness suite on HPC)
#
# ONE-TIME SETUP (do this first):
#   ssh-keygen -t ed25519          # accept defaults, no passphrase
#   ssh-copy-id cs1230459@hpc.iitd.ac.in
#
# If ssh-copy-id fails, do it manually:
#   cat ~/.ssh/id_ed25519.pub | ssh cs1230459@hpc.iitd.ac.in "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
#
# Without SSH keys, the script still works but prompts for
# password once per run (via SSH ControlMaster).
# ============================================================

set -uo pipefail

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    echo "Usage: ./test.sh <knn|kmeans|approx_knn|all>"
    echo "       ./test.sh fetch <knn|kmeans|approx_knn>   (download results from completed job)"
    echo "       ./test.sh correctness                     (run full test suite on HPC)"
    echo "       ./test.sh correctness --local             (run locally with sequential code)"
    echo "       ./test.sh fetch correctness               (download correctness results)"
    exit 1
fi

# ---- Configuration ----
HPC_USER="cs1230459"
HPC_HOST="hpc.iitd.ac.in"
HPC_DIR="Assignment_2"
PROJECT="col7880.cs1230459.course"
POLL_INTERVAL=5
TIMEOUT=900   # 15 min (includes queue wait)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- SSH ControlMaster (single password prompt per session) ----
SSH_CTRL="/tmp/ssh_hpc_a2"
SSH="ssh -o ControlMaster=auto -o ControlPath=${SSH_CTRL} -o ControlPersist=300 -o ServerAliveInterval=30"
SCP="scp -o ControlPath=${SSH_CTRL}"

# bash -l forces login shell so PATH includes PBS tools (qsub etc.)
run_hpc() { ${SSH} ${HPC_USER}@${HPC_HOST} "bash -l -c '$*'"; }

# ============================================================
# Helper: download correctness results from HPC
# ============================================================
fetch_correctness_results() {
    local HPC_TESTING_DIR="${HPC_DIR}/testing"
    local LOCAL_RESULTS="$SCRIPT_DIR/testing/tests/results"
    mkdir -p "$LOCAL_RESULTS"

    # Download log
    if ${SCP} ${HPC_USER}@${HPC_HOST}:~/${HPC_DIR}/correctness.log ./correctness.log 2>/dev/null; then
        echo "--- Build & Test Log ---"
        cat ./correctness.log
        echo ""
    else
        echo "No correctness.log found on HPC. Job may not have completed yet."
        echo "  Check: ssh ${HPC_USER}@${HPC_HOST} 'bash -l -c \"qstat -u ${HPC_USER}\"'"
        return 1
    fi

    # Download validation summary
    if ${SCP} ${HPC_USER}@${HPC_HOST}:~/${HPC_TESTING_DIR}/tests/outputs/validation_summary.txt "$LOCAL_RESULTS/validation_summary.txt" 2>/dev/null; then
        echo "--- Validation Summary ---"
        cat "$LOCAL_RESULTS/validation_summary.txt"
        echo ""
    else
        echo "(No validation_summary.txt found)"
    fi

    if ${SCP} ${HPC_USER}@${HPC_HOST}:~/${HPC_TESTING_DIR}/tests/outputs/run_summary.txt "$LOCAL_RESULTS/run_summary.txt" 2>/dev/null; then
        echo "--- Run Summary ---"
        cat "$LOCAL_RESULTS/run_summary.txt"
        echo ""
    else
        echo "(No run_summary.txt found)"
    fi

    # Download all per-test outputs
    echo "Downloading per-test outputs..."
    ${SCP} -r ${HPC_USER}@${HPC_HOST}:~/${HPC_TESTING_DIR}/tests/outputs/ "$LOCAL_RESULTS/outputs/" 2>/dev/null || true

    echo ""
    echo "=========================================="
    echo "  Results saved to: testing/tests/results/"
    echo "=========================================="
}

# ============================================================
# FETCH CORRECTNESS MODE
# ============================================================
if [[ "$MODE" == "fetch" && "${2:-}" == "correctness" ]]; then
    echo "=== Fetching Correctness Results ==="
    echo ""

    echo "Connecting to HPC..."
    run_hpc "true" || { echo "SSH connection failed."; exit 1; }

    fetch_correctness_results
    echo "Done!"
    exit 0
fi

# ============================================================
# CORRECTNESS LOCAL MODE
# ============================================================
if [[ "$MODE" == "correctness" && "${2:-}" == "--local" ]]; then
    echo "=== Local Correctness Test Suite (sequential) ==="
    echo ""

    TESTING_DIR="$SCRIPT_DIR/testing"
    INPUTS_DIR="$TESTING_DIR/tests/inputs"
    EXPECTED_DIR="$TESTING_DIR/tests/arinjay"
    LOCAL_RESULTS="$TESTING_DIR/tests/results"
    MAE_SCRIPT="$SCRIPT_DIR/mae_loss.py"
    SEQ_BINARY="$SCRIPT_DIR/seq"
    VALIDATION_SUMMARY="$LOCAL_RESULTS/validation_summary.txt"

    # Verify directories
    if [[ ! -d "$INPUTS_DIR" ]]; then
        echo "Error: testing/tests/inputs/ not found."
        exit 1
    fi
    if [[ ! -d "$EXPECTED_DIR" ]]; then
        echo "Error: testing/tests/arinjay/ (expected outputs) not found."
        exit 1
    fi
    if [[ ! -f "$MAE_SCRIPT" ]]; then
        echo "Error: mae_loss.py not found."
        exit 1
    fi

    # Build sequential binary
    echo "Building sequential binary..."
    g++ -O2 -o "$SEQ_BINARY" "$SCRIPT_DIR/submission.cpp" 2>&1
    if [[ $? -ne 0 ]]; then
        echo "Build FAILED."
        exit 1
    fi
    echo "  Build OK: $SEQ_BINARY"
    echo ""

    # Prepare results directory
    rm -rf "$LOCAL_RESULTS"
    mkdir -p "$LOCAL_RESULTS"

    # Header
    {
        printf "%-35s %-12s %-12s %-12s %s\n" "TEST" "KNN" "KMEANS" "APPROX_KNN" "NOTES"
        printf '%0.s-' {1..100}; echo
    } > "$VALIDATION_SUMMARY"

    echo "Running tests..."
    printf "  %-35s %-12s %-12s %-12s %s\n" "TEST" "KNN" "KMEANS" "APPROX_KNN" "NOTES"
    printf "  "; printf '%0.s-' {1..95}; echo

    TOTAL=0; V_PASS=0; V_FAIL=0

    for INPUT_FILE in "$INPUTS_DIR"/*.txt; do
        TEST_NAME="$(basename "$INPUT_FILE" .txt)"
        TEST_OUT_DIR="$LOCAL_RESULTS/$TEST_NAME"
        EXPECTED_TEST_DIR="$EXPECTED_DIR/$TEST_NAME"
        mkdir -p "$TEST_OUT_DIR"

        TOTAL=$((TOTAL + 1))
        KNN_VALID="x"; KMEANS_VALID="x"; APPROX_VALID="x"
        NOTES=""

        # Run knn
        if (cd "$TEST_OUT_DIR" && "$SEQ_BINARY" "$INPUT_FILE" knn) > /dev/null 2>&1; then
            mv -f "$TEST_OUT_DIR/knn_seq.txt" "$TEST_OUT_DIR/knn.txt" 2>/dev/null
            if [[ -f "$EXPECTED_TEST_DIR/knn.txt" ]] && diff -q "$TEST_OUT_DIR/knn.txt" "$EXPECTED_TEST_DIR/knn.txt" > /dev/null 2>&1; then
                KNN_VALID="PASS"
            else
                KNN_VALID="FAIL"
                NOTES+="knn-mismatch "
            fi
        else
            KNN_VALID="ERROR"
            NOTES+="knn-error "
        fi

        # Run kmeans
        if (cd "$TEST_OUT_DIR" && "$SEQ_BINARY" "$INPUT_FILE" kmeans) > /dev/null 2>&1; then
            mv -f "$TEST_OUT_DIR/kmeans_seq.txt" "$TEST_OUT_DIR/kmeans.txt" 2>/dev/null
            if [[ -f "$EXPECTED_TEST_DIR/kmeans.txt" ]] && diff -q "$TEST_OUT_DIR/kmeans.txt" "$EXPECTED_TEST_DIR/kmeans.txt" > /dev/null 2>&1; then
                KMEANS_VALID="PASS"
            else
                KMEANS_VALID="FAIL"
                NOTES+="kmeans-mismatch "
            fi
        else
            KMEANS_VALID="ERROR"
            NOTES+="kmeans-error "
        fi

        # Run approx_knn
        if (cd "$TEST_OUT_DIR" && "$SEQ_BINARY" "$INPUT_FILE" approx_knn) > /dev/null 2>&1; then
            mv -f "$TEST_OUT_DIR/approx_knn_seq.txt" "$TEST_OUT_DIR/approx_knn.txt" 2>/dev/null
            if [[ -f "$TEST_OUT_DIR/approx_knn.txt" && -f "$TEST_OUT_DIR/knn.txt" ]]; then
                MAE_OUTPUT=$(python3 "$MAE_SCRIPT" "$TEST_OUT_DIR/knn.txt" "$TEST_OUT_DIR/approx_knn.txt" 2>&1)
                MAE_VAL=$(echo "$MAE_OUTPUT" | grep "MAE:" | awk '{print $2}')
                if [[ -n "$MAE_VAL" ]]; then
                    MAE_CHECK=$(echo "$MAE_VAL < 3" | bc 2>/dev/null)
                    if [[ "$MAE_CHECK" -eq 1 ]]; then
                        APPROX_VALID="PASS($MAE_VAL)"
                    else
                        APPROX_VALID="FAIL($MAE_VAL)"
                        NOTES+="mae>=3 "
                    fi
                else
                    APPROX_VALID="ERROR"
                    NOTES+="mae-error "
                fi
            else
                APPROX_VALID="SKIP"
                NOTES+="missing-knn "
            fi
        else
            APPROX_VALID="ERROR"
            NOTES+="approx-error "
        fi

        # Tally
        if [[ "$KNN_VALID" == "PASS" && "$KMEANS_VALID" == "PASS" && "$APPROX_VALID" == PASS* ]]; then
            V_PASS=$((V_PASS + 1))
        else
            V_FAIL=$((V_FAIL + 1))
        fi

        printf "  %-35s %-12s %-12s %-12s %s\n" "$TEST_NAME" "$KNN_VALID" "$KMEANS_VALID" "$APPROX_VALID" "$NOTES"
        printf "%-35s %-12s %-12s %-12s %s\n" "$TEST_NAME" "$KNN_VALID" "$KMEANS_VALID" "$APPROX_VALID" "$NOTES" >> "$VALIDATION_SUMMARY"
    done

    echo ""
    echo "=========================================="
    echo "  Total: $TOTAL  Passed: $V_PASS  Failed: $V_FAIL"
    echo "  Results: testing/tests/results/"
    echo "  Summary: testing/tests/results/validation_summary.txt"
    echo "=========================================="

    if [[ $V_FAIL -gt 0 ]]; then
        exit 1
    fi
    exit 0
fi

# ============================================================
# CORRECTNESS HPC MODE
# ============================================================
if [[ "$MODE" == "correctness" ]]; then
    echo "=== Correctness Test Suite ==="
    echo ""

    TESTING_DIR="$SCRIPT_DIR/testing"
    HPC_TESTING_DIR="${HPC_DIR}/testing"

    # Verify local testing directory exists
    if [[ ! -d "$TESTING_DIR" ]]; then
        echo "Error: testing/ directory not found."
        exit 1
    fi

    echo "Connecting to HPC..."
    run_hpc "mkdir -p ~/${HPC_DIR}" || { echo "SSH connection failed."; exit 1; }

    # Upload source code + Makefile
    echo "Uploading source code..."
    ${SCP} Makefile submission.cu ${HPC_USER}@${HPC_HOST}:~/${HPC_DIR}/

    # Upload entire testing directory
    echo "Uploading testing suite..."
    run_hpc "rm -rf ~/${HPC_TESTING_DIR}"
    ${SCP} -r "$TESTING_DIR" ${HPC_USER}@${HPC_HOST}:~/${HPC_DIR}/

    # Also upload mae_loss.py to testing dir on HPC
    if [[ -f "$SCRIPT_DIR/mae_loss.py" ]]; then
        ${SCP} "$SCRIPT_DIR/mae_loss.py" ${HPC_USER}@${HPC_HOST}:~/${HPC_TESTING_DIR}/
    fi

    # Clean previous markers
    run_hpc "rm -f ~/${HPC_DIR}/correctness.done ~/${HPC_DIR}/correctness.log"

    # Create PBS script for correctness
    PBS_FILE="/tmp/a2_correctness.pbs"
    cat > "${PBS_FILE}" << 'PBSEOF'
#!/bin/bash
#PBS -N a2_correctness
#PBS -P col7880.cs1230459.course
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=00:15:00
#PBS -j oe

cd $HOME/Assignment_2
module purge
module load compiler/gcc/9.1.0
module load suite/nvidia-hpc-sdk/20.7/cuda11.0

{
    echo "=== BUILD ==="
    make
    BUILD_RC=$?
    if [ $BUILD_RC -ne 0 ]; then
        echo "BUILD FAILED (exit $BUILD_RC)"
    else
        echo "=== BUILD OK ==="
        echo ""

        # Copy the built binary into testing dir
        cp a2 testing/

        echo "=== RUNNING CORRECTNESS TESTS ==="
        cd testing
        bash run_tests.sh 2>&1
        echo "=== TESTS EXIT: $? ==="
    fi
} > correctness.log 2>&1

touch correctness.done
PBSEOF

    ${SCP} "${PBS_FILE}" ${HPC_USER}@${HPC_HOST}:~/${HPC_DIR}/correctness.pbs
    rm -f "${PBS_FILE}"

    # Submit
    echo "Submitting correctness job..."
    JOB_ID=$(run_hpc "cd ~/${HPC_DIR} && qsub correctness.pbs" 2>&1)
    if [[ $? -ne 0 ]]; then
        echo "qsub failed: ${JOB_ID}"
        exit 1
    fi
    echo "  Job ID: ${JOB_ID}"

    # Poll
    SECONDS=0
    while ! run_hpc "test -f ~/${HPC_DIR}/correctness.done" 2>/dev/null; do
        if (( SECONDS > TIMEOUT )); then
            echo ""
            echo "TIMEOUT after ${TIMEOUT}s. Job may still be queued."
            echo "  Fetch results later:  ./test.sh fetch correctness"
            echo "  Check: ssh ${HPC_USER}@${HPC_HOST} 'bash -l -c \"qstat -u ${HPC_USER}\"'"
            echo "  Kill:  ssh ${HPC_USER}@${HPC_HOST} 'bash -l -c \"qdel ${JOB_ID}\"'"
            exit 1
        fi
        printf "\r  Waiting... %ds" $SECONDS
        sleep ${POLL_INTERVAL}
    done
    printf "\r  Completed in %ds            \n" $SECONDS

    # ---- Download results ----
    echo ""
    fetch_correctness_results
    echo "Done!"
    exit 0
fi

# ============================================================
# ORIGINAL SINGLE-MODE LOGIC (knn / kmeans / approx_knn / all / fetch)
# ============================================================

# Handle "fetch" subcommand — just download results, no resubmit
FETCH_ONLY=false
if [[ "$MODE" == "fetch" ]]; then
    FETCH_ONLY=true
    MODE="${2:-}"
    if [[ -z "$MODE" ]]; then
        echo "Usage: ./test.sh fetch <knn|kmeans|approx_knn>"
        exit 1
    fi
fi

# Validate mode
MODES=()
if [[ "$MODE" == "all" ]]; then
    MODES=("knn" "kmeans" "approx_knn")
else
    if [[ "$MODE" != "knn" && "$MODE" != "kmeans" && "$MODE" != "approx_knn" ]]; then
        echo "Error: mode must be knn, kmeans, approx_knn, all, or correctness"
        exit 1
    fi
    MODES=("$MODE")
fi

# Check input.txt
if [[ ! -f "input.txt" ]]; then
    echo "Error: input.txt not found. Generate one with:"
    echo "  python3 dataset_generator.py 1000 10 20"
    exit 1
fi

echo "Connecting to HPC..."
run_hpc "mkdir -p ~/${HPC_DIR}" || { echo "SSH connection failed."; exit 1; }

if [[ "$FETCH_ONLY" == false ]]; then
    # Files to upload (add more source files here if needed)
    UPLOAD_FILES=(Makefile submission.cu input.txt)

    echo "Uploading: ${UPLOAD_FILES[*]}"
    ${SCP} "${UPLOAD_FILES[@]}" ${HPC_USER}@${HPC_HOST}:~/${HPC_DIR}/
fi

for CURRENT_MODE in "${MODES[@]}"; do
    echo ""
    echo "=========================================="
    if [[ "$FETCH_ONLY" == true ]]; then
        echo "  Fetching: ${CURRENT_MODE}"
    else
        echo "  Testing: ${CURRENT_MODE}"
    fi
    echo "=========================================="

    if [[ "$FETCH_ONLY" == false ]]; then
        # Clean previous run markers
        run_hpc "rm -f ~/${HPC_DIR}/job.done ~/${HPC_DIR}/job.log ~/${HPC_DIR}/${CURRENT_MODE}.txt"

        # Create PBS script locally and upload
        PBS_FILE="/tmp/a2_job_${CURRENT_MODE}.pbs"
        cat > "${PBS_FILE}" << PBSEOF
#!/bin/bash
#PBS -N a2_${CURRENT_MODE}
#PBS -P ${PROJECT}
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=00:10:00
#PBS -j oe

cd \$HOME/${HPC_DIR}
module purge
module load compiler/gcc/9.1.0
module load suite/nvidia-hpc-sdk/20.7/cuda11.0

{
    echo "=== BUILD ==="
    make
    BUILD_RC=\$?
    if [ \$BUILD_RC -ne 0 ]; then
        echo "BUILD FAILED (exit \$BUILD_RC)"
    else
        echo "=== RUN: ./a2 input.txt ${CURRENT_MODE} ==="
        time ./a2 input.txt ${CURRENT_MODE}
        echo "=== EXIT: \$? ==="
    fi
} > job.log 2>&1

touch job.done
PBSEOF

        ${SCP} "${PBS_FILE}" ${HPC_USER}@${HPC_HOST}:~/${HPC_DIR}/job.pbs
        rm -f "${PBS_FILE}"

        # Submit job
        echo "Submitting PBS job..."
        JOB_ID=$(run_hpc "cd ~/${HPC_DIR} && qsub job.pbs" 2>&1)
        if [[ $? -ne 0 ]]; then
            echo "qsub failed: ${JOB_ID}"
            continue
        fi
        echo "  Job ID: ${JOB_ID}"

        # Poll for completion
        SECONDS=0
        while ! run_hpc "test -f ~/${HPC_DIR}/job.done" 2>/dev/null; do
            if (( SECONDS > TIMEOUT )); then
                echo ""
                echo "TIMEOUT after ${TIMEOUT}s. Job may still be queued."
                echo "  Fetch results later:  ./test.sh fetch ${CURRENT_MODE}"
                echo "  Check manually: ssh ${HPC_USER}@${HPC_HOST} 'bash -l -c \"qstat -u ${HPC_USER}\"'"
                echo "  Kill job:       ssh ${HPC_USER}@${HPC_HOST} 'bash -l -c \"qdel ${JOB_ID}\"'"
                continue 2
            fi
            printf "\r  Waiting... %ds" $SECONDS
            sleep ${POLL_INTERVAL}
        done
        printf "\r  Completed in %ds            \n" $SECONDS
    fi

    # ---- Download results (runs for both normal and fetch mode) ----

    # Download and show log
    if ${SCP} ${HPC_USER}@${HPC_HOST}:~/${HPC_DIR}/job.log "./job_${CURRENT_MODE}.log" 2>/dev/null; then
        echo ""
        echo "--- Build & Run Log ---"
        cat "./job_${CURRENT_MODE}.log"
        echo ""
    else
        echo "No job.log found on HPC. Job may not have completed yet."
        echo "  Check: ssh ${HPC_USER}@${HPC_HOST} 'bash -l -c \"qstat -u ${HPC_USER}\"'"
        continue
    fi

    # Download output file
    if ${SCP} ${HPC_USER}@${HPC_HOST}:~/${HPC_DIR}/${CURRENT_MODE}.txt "./${CURRENT_MODE}.txt" 2>/dev/null; then
        LINES=$(wc -l < "./${CURRENT_MODE}.txt" | tr -d ' ')
        echo "${CURRENT_MODE}.txt downloaded (${LINES} lines)"
        echo "First 3 lines:"
        head -3 "./${CURRENT_MODE}.txt"
        echo "..."
    else
        echo "ERROR: ${CURRENT_MODE}.txt not found on HPC. Check log above."
        continue
    fi

    # Compare with sequential output
    # Approximate KNN should be evaluated against the exact KNN baseline.
    if [[ "${CURRENT_MODE}" == "approx_knn" ]]; then
        SEQ_FILE="./knn_seq.txt"
    else
        SEQ_FILE="./${CURRENT_MODE}_seq.txt"
    fi
    if [[ -f "$SEQ_FILE" ]]; then
        echo ""
        echo "--- Comparison with $(basename "$SEQ_FILE") ---"
        if diff -q "./${CURRENT_MODE}.txt" "$SEQ_FILE" > /dev/null 2>&1; then
            echo "PASS: Exact match!"
        else
            DIFF_LINES=$(diff "./${CURRENT_MODE}.txt" "$SEQ_FILE" | grep -c "^[<>]" || true)
            echo "DIFF: ${DIFF_LINES} lines differ. First differences:"
            diff "./${CURRENT_MODE}.txt" "$SEQ_FILE" | head -20
        fi

        # MAE loss
        if [[ -f "mae_loss.py" ]]; then
            echo ""
            echo "--- MAE Loss ---"
            python3 mae_loss.py "./${CURRENT_MODE}.txt" "$SEQ_FILE" 2>&1 || true
        fi
    else
        echo ""
        echo "(No ${CURRENT_MODE}_seq.txt found -- skipping comparison)"
    fi
done

echo ""
echo "Done!"
