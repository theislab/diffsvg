#!/bin/sh
# Runs every job line in results/jobs.txt, 4 at a time.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
xargs -P 4 -L 1 sh -c '../../.venv/bin/python run_cuprizone.py "$@" 2>&1 | grep -v Warn | tail -1' _ < results/jobs.txt > results/log.txt
