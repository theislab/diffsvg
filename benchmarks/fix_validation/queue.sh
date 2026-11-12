#!/bin/zsh
# usage: queue.sh <code> <exp>... ; runs sequentially, logs to results/<code>_<exp>.log
export DIFFSVG_OLD_SRC=~/Documents/GitHub/diffsvg/src DIFFSVG_LEGNINI=~/Documents/data/diffsvg/legnini23.h5ad DIFFSVG_XENIUM=~/Documents/data/diffsvg/both_conditions_5_7_without_noise.h5ad
code=$1; shift
mkdir -p results
for e in "$@"; do
  ../../.venv/bin/python run.py --code $code --exp $e --workers ${WORKERS:-6} > results/${code}_$e.log 2>&1 && tail -1 results/${code}_$e.log || echo "FAILED $code $e"
done
