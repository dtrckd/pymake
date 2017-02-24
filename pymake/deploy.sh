#!/bin/bash
PWD="/home/ama/adulac/workInProgress/networkofgraphs/process/pymake/pymake"
CMDS="roc_cmd"

NODES="nodeslist"
parallel  --sshloginfile $NODES --workdir $PWD  < $CMDS
