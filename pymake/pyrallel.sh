#!/bin/bash

CORES="2"

COMMAND="python3 fit.py -nv -w -i 200"

SPEC="HELP"
if [ ! -z "$1" ]; then
    SPEC="$1"
fi

if [ ! -z "$2" ]; then
    CORES="$2"
fi

./zymake.py runcmd $SPEC \
    | parallel --eta -k -j$CORES --colsep ' ' "$COMMAND {}"
