#!/bin/bash

USER='adulac'
HOST='racer'
REMOTE_LOC='/home/ama/adulac/workInProgress/networkofgraphs/process/pymake/data/'
LOCAL_LOC='../data/'

SPEC='EXPE_ICDM'
FTYPE='json'

TEMP_FILE='.pysync.out'
./zymake.py path $SPEC $FTYPE > $TEMP_FILE

#DR="--dry-run"
OPTS_RSYNC='-av -u  --modify-window=1 --stats --prune-empty-dirs -e ssh '
/usr/bin/rsync $DR \
    $OPTS_RSYNC \
    --include='*/' \
    --include-from="$TEMP_FILE" \
    --exclude '*' \
    ${USER}@${HOST}:"${REMOTE_LOC}" \
    "${LOCAL_LOC}"

rm -f $TEMP_FILE

