#!/bin/bash

# Script to run all benchmark baselines (no optimization) with logging
# Usage: ./run_all_baselines.sh [benchmark_name] [dataset_mode]
#   benchmark_name: all, hotpotqa, hover, aime, ifbench, livebench_math, papillon (default: all)
#   dataset_mode: lite, full, tiny, test (default: test)
#
# Note: When benchmark_name is "all", benchmarks run in PARALLEL
#
# Examples:
#   ./run_all_baselines.sh all test        # Run all benchmarks in parallel with test dataset
#   ./run_all_baselines.sh all lite        # Run all benchmarks in parallel with lite dataset
#   ./run_all_baselines.sh hotpotqa test   # Run single benchmark baseline
#   ./run_all_baselines.sh aime full       # Run AIME baseline with full dataset

set -e  # Exit on error

# Show help if requested
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 [benchmark_name] [dataset_mode]"
    echo ""
    echo "Arguments:"
    echo "  benchmark_name  Benchmark to run (default: all)"
    echo "                  Options: all, hotpotqa, hover, aime, ifbench, livebench_math, papillon"
    echo "                  Note: 'all' runs benchmarks in PARALLEL"
    echo "  dataset_mode    Dataset size (default: test)"
    echo "                  Options: lite, full, tiny, test"
    echo ""
    echo "Examples:"
    echo "  $0 all test               # Run all baselines in parallel with test dataset"
    echo "  $0 all lite               # Run all baselines in parallel with lite dataset"
    echo "  $0 hotpotqa test          # Run HotpotQA baseline"
    echo "  $0 aime full              # Run AIME baseline with full dataset"
    exit 0
fi

# Configuration
BENCHMARK="${1:-all}"
DATASET_MODE="${2:-test}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE="results/baseline_run_${TIMESTAMP}"

# Define benchmarks and their scripts
declare -A BENCHMARKS
BENCHMARKS=(
    ["hotpotqa"]="run_hotpotqa_trace.py"
    ["hover"]="run_hover_trace.py"
    ["aime"]="run_aime_trace.py"
    ["ifbench"]="run_ifbench_trace.py"
    ["livebench_math"]="run_livebench_math_trace.py"
    ["papillon"]="run_papillon_trace.py"
)

# Function to run a single benchmark baseline
run_baseline() {
    local bench_name=$1
    local script_name=$2
    
    echo "========================================"
    echo "Starting baseline: ${bench_name}"
    echo "Dataset mode: ${DATASET_MODE}"
    echo "Mode: BASELINE ONLY (no optimization)"
    echo "Timestamp: ${TIMESTAMP}"
    echo "========================================"
    
    # Create log directory
    local logdir="${RESULTS_BASE}/${bench_name}_Baseline"
    mkdir -p "$logdir"
    
    # Copy the script to log directory
    cp "$script_name" "$logdir/"
    
    # Run the baseline and redirect output to log
    echo "Running: python -u ${script_name} --dataset-mode ${DATASET_MODE} --run-test-baseline"
    python -u "$script_name" --dataset-mode "$DATASET_MODE" --run-test-baseline > "$logdir/output.log" 2>&1
    
    # Extract and save test results
    echo "Extracting test results..."
    grep -A 10 "Test Results:" "$logdir/output.log" > "$logdir/test_results.txt" 2>/dev/null || \
    grep -A 10 "Running Parallel Evaluation" "$logdir/output.log" >> "$logdir/test_results.txt" 2>/dev/null || \
    echo "No test results found in output" > "$logdir/test_results.txt"
    
    echo "✓ Completed: ${bench_name}"
    echo "  Log directory: ${logdir}"
    echo "  Output log: ${logdir}/output.log"
    echo "  Test results: ${logdir}/test_results.txt"
    echo ""
}

# Main execution
mkdir -p "$RESULTS_BASE"

# Create a summary file
SUMMARY_FILE="${RESULTS_BASE}/summary.txt"
echo "Baseline Evaluation Summary" > "$SUMMARY_FILE"
echo "Timestamp: ${TIMESTAMP}" >> "$SUMMARY_FILE"
echo "Dataset Mode: ${DATASET_MODE}" >> "$SUMMARY_FILE"
echo "======================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

if [ "$BENCHMARK" = "all" ]; then
    echo "Running ALL benchmark baselines..."
    echo "Mode: PARALLEL execution"
    echo "========================================"
    
    # Store PIDs for parallel execution
    declare -a PIDS
    
    # Run all benchmarks in parallel
    for bench_name in "${!BENCHMARKS[@]}"; do
        script_name="${BENCHMARKS[$bench_name]}"
        if [ -f "$script_name" ]; then
            # Run in background for parallel execution
            run_baseline "$bench_name" "$script_name" &
            PIDS+=($!)
            echo "Started ${bench_name} baseline in background (PID: $!)"
        else
            echo "Warning: Script not found: $script_name"
        fi
    done
    
    # Wait for all parallel jobs to complete
    echo ""
    echo "========================================"
    echo "Waiting for all parallel jobs to complete..."
    echo "Running PIDs: ${PIDS[@]}"
    echo "========================================"
    
    for pid in "${PIDS[@]}"; do
        wait $pid
        echo "Process $pid completed"
    done
    
    echo ""
    echo "========================================"
    echo "All baseline evaluations completed!"
    echo "Results saved in: ${RESULTS_BASE}/"
    echo "========================================"
    echo ""
    
    # Generate summary of results
    echo "Generating summary of all results..."
    echo "" >> "$SUMMARY_FILE"
    
    for bench_name in "${!BENCHMARKS[@]}"; do
        local logdir="${RESULTS_BASE}/${bench_name}_Baseline"
        echo "=== ${bench_name} ===" >> "$SUMMARY_FILE"
        if [ -f "$logdir/test_results.txt" ]; then
            cat "$logdir/test_results.txt" >> "$SUMMARY_FILE"
        else
            echo "Results not found" >> "$SUMMARY_FILE"
        fi
        echo "" >> "$SUMMARY_FILE"
    done
    
    echo "Summary saved to: ${SUMMARY_FILE}"
    echo ""
    echo "Quick summary:"
    cat "$SUMMARY_FILE"
    
else
    # Run specific benchmark
    if [[ -v BENCHMARKS[$BENCHMARK] ]]; then
        script_name="${BENCHMARKS[$BENCHMARK]}"
        if [ -f "$script_name" ]; then
            run_baseline "$BENCHMARK" "$script_name"
            
            # Add to summary
            echo "=== ${BENCHMARK} ===" >> "$SUMMARY_FILE"
            local logdir="${RESULTS_BASE}/${BENCHMARK}_Baseline"
            if [ -f "$logdir/test_results.txt" ]; then
                cat "$logdir/test_results.txt" >> "$SUMMARY_FILE"
            fi
            
            echo ""
            cat "$SUMMARY_FILE"
        else
            echo "Error: Script not found: $script_name"
            exit 1
        fi
    else
        echo "Error: Unknown benchmark: $BENCHMARK"
        echo "Available benchmarks: ${!BENCHMARKS[@]} all"
        exit 1
    fi
fi

