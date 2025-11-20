#!/bin/bash

# Script to run all benchmark traces with logging
# Usage: ./run_all_benchmarks.sh [benchmark_name] [dataset_mode] [optimizer]
#   benchmark_name: all, hotpotqa, hover, aime, ifbench, livebench_math, papillon (default: all)
#   dataset_mode: lite, full, tiny, test (default: lite)
#   optimizer: OptoPrime, TextGrad (default: OptoPrime)
#
# Note: When benchmark_name is "all", benchmarks run in PARALLEL
#
# Examples:
#   ./run_all_benchmarks.sh all lite OptoPrime        # Run all benchmarks in parallel with OptoPrime
#   ./run_all_benchmarks.sh all lite TextGrad         # Run all benchmarks in parallel with TextGrad
#   ./run_all_benchmarks.sh hotpotqa lite TextGrad    # Run single benchmark with TextGrad
#   ./run_all_benchmarks.sh aime full OptoPrime       # Run AIME with full dataset

set -e  # Exit on error

# Show help if requested
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 [benchmark_name] [dataset_mode] [optimizer]"
    echo ""
    echo "Arguments:"
    echo "  benchmark_name  Benchmark to run (default: all)"
    echo "                  Options: all, hotpotqa, hover, aime, ifbench, livebench_math, papillon"
    echo "                  Note: 'all' runs benchmarks in PARALLEL"
    echo "  dataset_mode    Dataset size (default: lite)"
    echo "                  Options: lite, full, tiny, test"
    echo "  optimizer       Optimizer to use (default: OptoPrime)"
    echo "                  Options: OptoPrime, TextGrad"
    echo ""
    echo "Examples:"
    echo "  $0 all lite OptoPrime         # Run all in parallel with OptoPrime"
    echo "  $0 all lite TextGrad          # Run all in parallel with TextGrad"
    echo "  $0 hotpotqa lite TextGrad     # Run HotpotQA with TextGrad"
    echo "  $0 aime full OptoPrime        # Run AIME with full dataset"
    exit 0
fi

# Configuration
BENCHMARK="${1:-all}"
DATASET_MODE="${2:-lite}"
OPTIMIZER="${3:-OptoPrime}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE="results/run_${TIMESTAMP}"

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

# Define number of steps for each benchmark
# Calculated as: (known_max_calls / 10 + 100) rounded to nearest 100
declare -A NUM_STEPS
NUM_STEPS=(
    ["aime"]=300
    ["ifbench"]=500
    ["livebench_math"]=300
    ["papillon"]=300
    ["hotpotqa"]=800      
    ["hover"]=800       
)

# Function to run a single benchmark
run_benchmark() {
    local bench_name=$1
    local script_name=$2
    local num_steps=$3
    local optimizer=$4
    
    echo "========================================"
    echo "Starting benchmark: ${bench_name}"
    echo "Dataset mode: ${DATASET_MODE}"
    echo "Number of steps: ${num_steps}"
    echo "Optimizer: ${optimizer}"
    echo "Timestamp: ${TIMESTAMP}"
    echo "========================================"
    
    # Create log directory
    local logdir="${RESULTS_BASE}/${bench_name}_${optimizer}"
    mkdir -p "$logdir"
    
    # Copy the script to log directory
    cp "$script_name" "$logdir/"
    
    # Run the benchmark and redirect output to log
    echo "Running: python -u ${script_name} --dataset-mode ${DATASET_MODE} --num-steps ${num_steps} --optimizer ${optimizer} --save-dir ${logdir}"
    python -u "$script_name" --dataset-mode "$DATASET_MODE" --num-steps "$num_steps" --optimizer "$optimizer" --save-dir "$logdir" > "$logdir/output.log" 2>&1
    
    # Extract and save test results
    echo "Extracting test results..."
    grep -A 10 "Test Results:" "$logdir/output.log" > "$logdir/test_results.txt" 2>/dev/null || \
    grep -A 10 "Final Parallel Evaluation" "$logdir/output.log" >> "$logdir/test_results.txt" 2>/dev/null || \
    echo "No test results found in output" > "$logdir/test_results.txt"
    
    echo "✓ Completed: ${bench_name}"
    echo "  Log directory: ${logdir}"
    echo "  Output log: ${logdir}/output.log"
    echo "  Test results: ${logdir}/test_results.txt"
    echo ""
}

# Main execution
mkdir -p "$RESULTS_BASE"

if [ "$BENCHMARK" = "all" ]; then
    echo "Running ALL benchmarks with optimizer: ${OPTIMIZER}..."
    echo "Mode: PARALLEL execution"
    echo "========================================"
    
    # Store PIDs for parallel execution
    declare -a PIDS
    
    # Run all benchmarks in parallel
    for bench_name in "${!BENCHMARKS[@]}"; do
        script_name="${BENCHMARKS[$bench_name]}"
        num_steps="${NUM_STEPS[$bench_name]}"
        if [ -f "$script_name" ]; then
            # Run in background for parallel execution
            run_benchmark "$bench_name" "$script_name" "$num_steps" "$OPTIMIZER" &
            PIDS+=($!)
            echo "Started ${bench_name} in background (PID: $!)"
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
    
    echo "========================================"
    echo "All benchmarks completed!"
    echo "Results saved in: ${RESULTS_BASE}/"
    echo "========================================"
else
    # Run specific benchmark
    if [[ -v BENCHMARKS[$BENCHMARK] ]]; then
        script_name="${BENCHMARKS[$BENCHMARK]}"
        num_steps="${NUM_STEPS[$BENCHMARK]}"
        if [ -f "$script_name" ]; then
            run_benchmark "$BENCHMARK" "$script_name" "$num_steps" "$OPTIMIZER"
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

