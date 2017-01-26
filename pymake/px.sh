#!/bin/bash

#############
### GNU Parallel parameters
JOBS="3"

COMMAND="python ./fit.py -w -i 200 --refdir debug"

#############
### parameters
CORPUS="clique4"
MODELS="mmsb_cgs ibp"
Ks="5"
ALPHAS="auto"
Ns="100"
homo="0 2"
RUNS=""

for alpha in $ALPHAS; do
    for hom in $homo; do
        for corpus in $CORPUS; do
            for N in $Ns; do
                for K in $Ks; do
                    for model in $MODELS; do
                        RUNS="${RUNS} -m $model --homo $hom -k $K --alpha $alpha -c $corpus -n $N\n"
                    done
                done
            done
        done
    done
done
# Remove last breakline
RUNS=${RUNS::-2}

###--- Gnu Parallel ---###
#parallel --no-notice -k -j$JOBS  $RUN ::: {1..4}
echo -e "$RUNS" | parallel --delay 1 --load 42 --eta -k -j$JOBS --colsep ' ' "$COMMAND {}"
