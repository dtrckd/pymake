#!/bin/bash

Engine='circo sfdp'
#Engine='circo dot fdp neato nop nop1 nop2 osage patchwork sfdp twopi'

for e in $Engine; do
    for f in graph/*.dot; do
        echo "$f -> $e"
        dot -K$e -Tpng  $f > ${f%%.dot}_$e.png
    done
done

