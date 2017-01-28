#!/bin/bash

#############
### GNU Parallel parameters
JOBS="3"

COMMAND="python3 ./fit.py -nv -w -i 200 --refdir debug"

#############
### parameters
CORPUS="nips12 kos nips reuter50 20ngroups" 
#CORPUS="generator5 generator6"
MODELS="lda_vb"
Ks="20"
ALPHAS="auto"
Ns="all"
homo="2"
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
#parallel --delay 10 ./generate.py -w -k {}  ::: $(echo 5 10 15 20)
echo -e "$RUNS" | parallel --delay 1 --load 42 --eta -k -j$JOBS --colsep ' ' "$COMMAND {}"
