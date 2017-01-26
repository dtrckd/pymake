#!/bin/bash

#############
### GNU Parallel parameters
JOBS="2"

#############
### LDA parameters
COMMAND="./lda_run.py -p"
MODELS="ldamodel ldafullbaye"
Ks="6 20"
#ALPHAS="symmetric auto"
ALPHAS="asymmetric"
#Ns="500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11314"
#Ns="500 1000 2000 3000 4000 5000 6000 7000"
Ns="500 1000 2000 3000"
RUNS=""

for N in $Ns; do
    for K in $Ks; do
        for alpha in $ALPHAS; do
            for model in $MODELS; do
                RUNS="${RUNS} -m $model -k $K --alpha $alpha -n $N\n"
            done
        done
    done
done
# Remove last breakline
RUNS=${RUNS::-2}

###--- Gnu Parallel ---###
#parallel --no-notice -k -j$JOBS  $RUN ::: {1..4}
echo -e  "$RUNS" | parallel -k -j$JOBS --colsep ' ' "$COMMAND {}"
