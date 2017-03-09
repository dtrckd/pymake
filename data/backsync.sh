#!/bin/bash

#SSH='adulac@tiger'
#IN="/home/ama/adulac/workInProgress/networkofgraphs/process/pymake/data/"

SSH='dulac@pitmanyor'
IN="/home/dulac/Desktop/workInProgress/networkofgraphs/process/pymake/data/"

OUT="./"
T="networks"

Exclude="debug111111"

SIMUL="-n"
OPTS="--update"

if [ "$1" == "-f" ]; then
    SIMUL=""; fi

#rsync $SIMUL  -av -u --modify-window=2 --stats -m $OPTS \
rsync $SIMUL $OPTS -vah --stats -m --exclude $Exclude \
    -e ssh  ${SSH}:${IN}/$T/ ${OUT}/$T

echo
echo "rsync $SIMUL $OPTS -vah --stats -m -e ssh --exclude $Exclude  ${SSH}:${IN}/$T/ ${OUT}/$T"

###
#rsync --dry-run  -av -u --modify-window=2  --stats --prune-empty-dirs  -e ssh --include '*/'  --include='debug/***' --exclude='*'  ./ dulac@pitmanyor:/home/dulac/ddebug
#rsync --dry-run  -av -u --modify-window=2 --stats --prune-empty-dirs  -e ssh    adulac@racer:/home/ama/adulac/workInProgress/networkofgraphs/process/PyNPB/data/networks/ ./


