#!/bin/bash

Engine='circo dot fdp neato nop nop1 nop2 osage patchwork sfdp twopi'

for e in $Engine; do
    echo "doting $e"
    dot -K$e -Tpng  graph.dot > data/plot/graph_$e.png
done

cp data/plot/graph_dot.png .
ristretto data/plot/graph_dot.png
