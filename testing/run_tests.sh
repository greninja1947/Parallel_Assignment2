#!/usr/bin/env bash
# ============================================================
# run_tests.sh
# ============================================================
# Builds the project, runs all test inputs through knn,
# approx_knn, and kmeans, validates against expected outputs,
# and computes MAE(knn, approx_knn) for every test case.
#
# Usage (from the a2/ directory):
#   bash testing/run_tests.sh
#
# Validation Criteria:
#   - KNN/KMeans: Must match expected output exactly
#   - approx_KNN: MAE vs KNN must be < 3
#
# Outputs:
#   tests/outputs/<test_name>/knn.txt
#   tests/outputs/<test_name>/approx_knn.txt
#   tests/outputs/<test_name>/kmeans.txt
#   tests/outputs/mae_summary.txt
#   tests/outputs/validation_summary.txt
# ============================================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUTS_DIR="$SCRIPT_DIR/tests/inputs"
OUTPUTS_DIR="$SCRIPT_DIR/tests/outputs"
EXPECTED_DIR="$SCRIPT_DIR/tests/arinjay"
BINARY=""  # resolved in Step 1
MAE_SCRIPT="$SCRIPT_DIR/mae_loss.py"
MAE_SUMMARY="$OUTPUTS_DIR/mae_summary.txt"
RUN_SUMMARY="$OUTPUTS_DIR/run_summary.txt"
VALIDATION_SUMMARY="$OUTPUTS_DIR/validation_summary.txt"

mkdir -p "$OUTPUTS_DIR"

# Copy mae_loss.py if it doesn't exist
if [[ ! -f "$MAE_SCRIPT" ]]; then
    if [[ -f "$SCRIPT_DIR/../mae_loss.py" ]]; then
        cp "$SCRIPT_DIR/../mae_loss.py" "$MAE_SCRIPT"
    else
        echo "ERROR: mae_loss.py not found. Cannot compute MAE."
        exit 1
    fi
fi

# Check expected outputs directory
if [[ ! -d "$EXPECTED_DIR" ]]; then
    echo "ERROR: Expected outputs directory '$EXPECTED_DIR' not found."
    echo "Place correct outputs in tests/arinjay/<test_name>/{knn,kmeans,approx_knn}.txt"
    exit 1
fi

# ── Colours (graceful degradation if not a terminal) ──────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
if [ ! -t 1 ]; then RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; RESET=''; fi

# ── Step 1 : Find or build binary ─────────────────────────────
echo -e "${BOLD}${CYAN}=== Step 1: Locating binary ===${RESET}"

# Check multiple locations: testing/a2 (copied by PBS), or ../a2 (project root)
if [[ -x "$SCRIPT_DIR/a2" ]]; then
    BINARY="$SCRIPT_DIR/a2"
    echo -e "${GREEN}  Found binary: $BINARY${RESET}"
elif [[ -x "$SCRIPT_DIR/../a2" ]]; then
    BINARY="$SCRIPT_DIR/../a2"
    echo -e "${GREEN}  Found binary: $BINARY${RESET}"
else
    # Try to build from project root
    echo "  Binary not found, building..."
    if make -C "$SCRIPT_DIR/.." 2>&1; then
        BINARY="$SCRIPT_DIR/../a2"
        echo -e "${GREEN}  Build succeeded.${RESET}"
    else
        echo -e "${RED}  Build FAILED. Fix compilation errors before running tests.${RESET}"
        exit 1
    fi
fi
echo ""

# ── Step 2 : Run tests ────────────────────────────────────────
echo -e "${BOLD}${CYAN}=== Step 2: Running tests ===${RESET}"

# Collect all input files, sorted
mapfile -t INPUT_FILES < <(ls "$INPUTS_DIR"/*.txt 2>/dev/null | sort)

if [ ${#INPUT_FILES[@]} -eq 0 ]; then
    echo -e "${RED}No test input files found in $INPUTS_DIR.${RESET}"
    echo "Run:  python3 generate_tests.py  first."
    exit 1
fi

PASS=0; FAIL=0
declare -A TEST_STATUS        # test_name -> PASS|FAIL
declare -A SKIP_MAE           # test_name -> 1 if MAE should be skipped

# Header for run summary
{
    printf "%-35s %-10s %-10s %-10s %s\n" "TEST" "KNN" "APPROX_KNN" "KMEANS" "NOTES"
    printf '%0.s-' {1..90}; echo
} > "$RUN_SUMMARY"

# Header for validation summary
{
    printf "%-35s %-12s %-12s %-12s %s\n" "TEST" "KNN" "KMEANS" "APPROX_KNN" "NOTES"
    printf '%0.s-' {1..100}; echo
} > "$VALIDATION_SUMMARY"

for INPUT_FILE in "${INPUT_FILES[@]}"; do
    TEST_NAME="$(basename "$INPUT_FILE" .txt)"
    TEST_OUT_DIR="$OUTPUTS_DIR/$TEST_NAME"
    EXPECTED_TEST_DIR="$EXPECTED_DIR/$TEST_NAME"
    mkdir -p "$TEST_OUT_DIR"

    KNN_OUT="$TEST_OUT_DIR/knn.txt"
    APPROX_OUT="$TEST_OUT_DIR/approx_knn.txt"
    KMEANS_OUT="$TEST_OUT_DIR/kmeans.txt"

    KNN_EXPECTED="$EXPECTED_TEST_DIR/knn.txt"
    APPROX_EXPECTED="$EXPECTED_TEST_DIR/approx_knn.txt"
    KMEANS_EXPECTED="$EXPECTED_TEST_DIR/kmeans.txt"

    STATUS_KNN="PASS"; STATUS_APPROX="PASS"; STATUS_KMEANS="PASS"
    NOTES=""
    ANY_FAIL=0

    printf "  %-35s " "$TEST_NAME"

    # Run from project root so output files land in a known place
    PROJ_ROOT="$(cd "$(dirname "$BINARY")" && pwd)"

    # -- knn --
    if (cd "$PROJ_ROOT" && "$BINARY" "$INPUT_FILE" knn) > /dev/null 2>&1; then
        mv -f "$PROJ_ROOT/knn.txt" "$KNN_OUT"
    else
        STATUS_KNN="FAIL"; ANY_FAIL=1
        NOTES+="knn-error "
    fi

    # -- approx_knn --
    if (cd "$PROJ_ROOT" && "$BINARY" "$INPUT_FILE" approx_knn) > /dev/null 2>&1; then
        mv -f "$PROJ_ROOT/approx_knn.txt" "$APPROX_OUT"
    else
        STATUS_APPROX="FAIL"; ANY_FAIL=1
        NOTES+="approx_knn-error "
    fi

    # -- kmeans --
    if (cd "$PROJ_ROOT" && "$BINARY" "$INPUT_FILE" kmeans) > /dev/null 2>&1; then
        mv -f "$PROJ_ROOT/kmeans.txt" "$KMEANS_OUT"
    else
        STATUS_KMEANS="FAIL"; ANY_FAIL=1
        NOTES+="kmeans-error "
    fi

    if [ $ANY_FAIL -eq 0 ]; then
        echo -e "${GREEN}PASS${RESET}"
        TEST_STATUS["$TEST_NAME"]="PASS"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${RESET} ($NOTES)"
        TEST_STATUS["$TEST_NAME"]="FAIL"
        SKIP_MAE["$TEST_NAME"]=1
        FAIL=$((FAIL + 1))
    fi

    printf "%-35s %-10s %-10s %-10s %s\n" \
        "$TEST_NAME" "$STATUS_KNN" "$STATUS_APPROX" "$STATUS_KMEANS" "$NOTES" \
        >> "$RUN_SUMMARY"
done

echo ""
echo -e "  Results: ${GREEN}${PASS} passed${RESET}  ${RED}${FAIL} failed${RESET}"
echo ""

# ── Step 3 : Validation (comparing against expected outputs) ──
echo -e "${BOLD}${CYAN}=== Step 3: Validating outputs ===${RESET}"

printf "  %-35s %-12s %-12s %-12s %s\n" "TEST" "KNN" "KMEANS" "APPROX_KNN" "NOTES"
printf "  "; printf '%0.s-' {1..95}; echo

VALIDATION_PASS=0
VALIDATION_FAIL=0

for INPUT_FILE in "${INPUT_FILES[@]}"; do
    TEST_NAME="$(basename "$INPUT_FILE" .txt)"
    TEST_OUT_DIR="$OUTPUTS_DIR/$TEST_NAME"
    EXPECTED_TEST_DIR="$EXPECTED_DIR/$TEST_NAME"

    KNN_OUT="$TEST_OUT_DIR/knn.txt"
    APPROX_OUT="$TEST_OUT_DIR/approx_knn.txt"
    KMEANS_OUT="$TEST_OUT_DIR/kmeans.txt"

    KNN_EXPECTED="$EXPECTED_TEST_DIR/knn.txt"
    KMEANS_EXPECTED="$EXPECTED_TEST_DIR/kmeans.txt"

    VALIDATION_NOTES=""
    KNN_VALID="✗"; KMEANS_VALID="✗"; APPROX_VALID="✗"

    # Skip if run failed
    if [ "${SKIP_MAE[$TEST_NAME]+_}" ]; then
        printf "  %-35s %-12s %-12s %-12s %s\n" "$TEST_NAME" "RUN_ERROR" "RUN_ERROR" "RUN_ERROR" "execution failed"
        VALIDATION_FAIL=$((VALIDATION_FAIL + 1))
        printf "%-35s %-12s %-12s %-12s %s\n" "$TEST_NAME" "RUN_ERROR" "RUN_ERROR" "RUN_ERROR" "execution failed" >> "$VALIDATION_SUMMARY"
        continue
    fi

    # Validate KNN
    if [ -f "$KNN_EXPECTED" ] && [ -f "$KNN_OUT" ]; then
        if diff -q "$KNN_OUT" "$KNN_EXPECTED" > /dev/null 2>&1; then
            KNN_VALID="✓"
        else
            VALIDATION_NOTES+="knn-mismatch "
        fi
    fi

    # Validate KMeans
    if [ -f "$KMEANS_EXPECTED" ] && [ -f "$KMEANS_OUT" ]; then
        if diff -q "$KMEANS_OUT" "$KMEANS_EXPECTED" > /dev/null 2>&1; then
            KMEANS_VALID="✓"
        else
            VALIDATION_NOTES+="kmeans-mismatch "
        fi
    fi

    # Validate approx_KNN (MAE < 3)
    if [ -f "$APPROX_OUT" ] && [ -f "$KNN_OUT" ]; then
        MAE_OUTPUT=$(python3 "$MAE_SCRIPT" "$KNN_OUT" "$APPROX_OUT" 2>&1)
        MAE_VAL=$(echo "$MAE_OUTPUT" | grep "MAE:" | awk '{print $2}')
        if [ -z "$MAE_VAL" ]; then
            VALIDATION_NOTES+="mae-error "
        else
            # Check if MAE < 3
            MAE_CHECK=$(echo "$MAE_VAL < 3" | bc 2>/dev/null)
            if [ "$MAE_CHECK" -eq 1 ]; then
                APPROX_VALID="✓ ($MAE_VAL)"
            else
                VALIDATION_NOTES+="mae-exceed($MAE_VAL) "
            fi
        fi
    fi

    if [[ "$KNN_VALID" == "✓" && "$KMEANS_VALID" == "✓" && "$APPROX_VALID" =~ "✓" ]]; then
        VALIDATION_PASS=$((VALIDATION_PASS + 1))
    else
        VALIDATION_FAIL=$((VALIDATION_FAIL + 1))
    fi

    printf "  %-35s %-12s %-12s %-12s %s\n" "$TEST_NAME" "$KNN_VALID" "$KMEANS_VALID" "$APPROX_VALID" "$VALIDATION_NOTES"
    printf "%-35s %-12s %-12s %-12s %s\n" "$TEST_NAME" "$KNN_VALID" "$KMEANS_VALID" "$APPROX_VALID" "$VALIDATION_NOTES" >> "$VALIDATION_SUMMARY"
done

echo ""

# ── Step 4 : Summary ──────────────────────────────────────────
echo -e "${BOLD}${CYAN}=== Step 4: Summary ===${RESET}"
TOTAL=${#INPUT_FILES[@]}
echo -e "  Total tests      : ${BOLD}${TOTAL}${RESET}"
echo -e "  Build/Run        : ${GREEN}${PASS}${RESET} passed  ${RED}${FAIL}${RESET} failed"
echo -e "  Validation       : ${GREEN}${VALIDATION_PASS}${RESET} passed  ${RED}${VALIDATION_FAIL}${RESET} failed"
echo ""
echo -e "  Run summary          : ${BOLD}tests/outputs/run_summary.txt${RESET}"
echo -e "  Validation report    : ${BOLD}tests/outputs/validation_summary.txt${RESET}"
echo -e "  Per-test outputs     : ${BOLD}tests/outputs/<test_name>/{knn,approx_knn,kmeans}.txt${RESET}"
echo ""

if [ $FAIL -gt 0 ] || [ $VALIDATION_FAIL -gt 0 ]; then
    echo -e "${YELLOW}  ⚠ Some tests failed. See validation_summary.txt${RESET}"
    exit 1
fi

echo -e "${GREEN}  ✓ All tests passed and validated!${RESET}"
