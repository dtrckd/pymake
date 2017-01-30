#!/bin/bash

JOBS="2"

COMMAND="python3 fit.py -nv -w -i 200"

SPEC="RUN_DD"
if [ ! -z "$1" ]; then
    SPEC="$1"
fi

./zymake.py runcmd $SPEC | parallel --eta -k -j$JOBS --colsep ' ' "$COMMAND {}"
