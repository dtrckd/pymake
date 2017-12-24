#!/bin/bash

SSH='adulac@tiger'
IN="/home/ama/adulac/workInProgress/networkofgraphs/process/pymake/data/"

#SSH='dulac@pitmanyor'
#IN="/home/dulac/Desktop/workInProgress/networkofgraphs/process/pymake/data/"

OUT="./"
T="networks"

#FILTER="--exclude debug111111"
#FILTER='--include "*/" --include "**pnas[23]**" --exclude "*"'
#FILTER='--include "*/" --include "*scvb***"  --exclude "*"'
FILTER='--include "*/" --include "*noel***"  --exclude "*"'
#FILTER=

SIMUL="-n"
OPTS="--update"

if [ "$1" == "-f" ]; then
    SIMUL=""; fi

#rsync $SIMUL  -av -u --modify-window=2 --stats -m $OPTS \
eval rsync $SIMUL $OPTS -vah --stats -m $FILTER \
    -e ssh  ${SSH}:${IN}/$T/ ${OUT}/$T

echo
echo "rsync $SIMUL $OPTS -vah --stats -m $FILTER -e ssh  ${SSH}:${IN}/$T/ ${OUT}/$T"

###
#rsync --dry-run  -av -u --modify-window=2  --stats --prune-empty-dirs  -e ssh --include '*/'  --include='debug/***' --exclude='*'  ./ dulac@pitmanyor:/home/dulac/ddebug
#rsync --dry-run  -av -u --modify-window=2 --stats --prune-empty-dirs  -e ssh    adulac@racer:/home/ama/adulac/workInProgress/networkofgraphs/process/PyNPB/data/networks/ ./


