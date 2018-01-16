#!/bin/bash

JOBS="2"

COMMAND="./lda_run.py"
RUNS="-m ldamodel -k 6 --alpha asymmetric -n 500  \n
      -m ldafullbaye -k 6 --alpha asymmetric -n 500"

#parallel --no-notice -k -j$JOBS  $RUN ::: {1..4}
echo "$RUNS" | parallel --no-notice -k -j$JOBS --colsep ' ' "$COMMAND {}"
