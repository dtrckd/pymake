#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

### Command
# files=$(find networks/generator/ -type f | grep -iEv "(.json|.pk)" | grep -iE "(debug10)" | xargs wc)

max_iter = 201

m = []
for line in sys.stdin:
    m.append(line.strip())

m = m[:-1]
tot_run = len(m)
tot_iter = len(m) * max_iter
iterat = sum( [ int(l.split(' ')[0]) for l in m] )
progress =  iterat / tot_iter

print('inference process... %s' % progress)



