#!/bin/bash

DEBUG="debug3"

IN="/home/ama/adulac/workInProgress/networkofgraphs/process/PyNPB/data/networks/"
OUT="./"
SSH='adulac@tiger'
SIMUL="-n"
if [ "$1" == "-f" ]; then
    SIMUL=""; fi

#rsync $SIMUL  -av -u --modify-window=2 --stats -m $OPTS \
rsync $SIMUL  -av --stats -m $OPTS \
    -e ssh  $SSH:$IN $OUT

###
#rsync --dry-run  -av -u --modify-window=2  --stats --prune-empty-dirs  -e ssh --include '*/'  --include='debug/***' --exclude='*'  ./ dulac@pitmanyor:/home/dulac/ddebug
#rsync --dry-run  -av -u --modify-window=2 --stats --prune-empty-dirs  -e ssh    adulac@racer:/home/ama/adulac/workInProgress/networkofgraphs/process/PyNPB/data/networks/ ./


