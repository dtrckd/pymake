#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
from tabulate import tabulate

from pymake import ExpTensor, OrderedDict
from frontend.frontend_io import *
from util.utils import *
from util.argparser import argparser
from pymake.expe.spec import _spec

lgg = logging.getLogger('root')

USAGE = '''\
# Usage:
    expe_k [model]
'''

#TENSOR = _spec.EXPE_ICDM_R
TENSOR = _spec.EXPE_ICDM_R_R

expe_args = argparser.exp_tabulate(USAGE)

###################################################################
# Data Forest config
map_parameters = TENSOR
### Seek experiments results
target_files = make_forest_path(map_parameters, 'json')
### Make Tensor Forest of results
rez = forest_tensor(target_files, map_parameters)

###################################################################
# Experimentation
#

### Expe 1 settings
# debug10, immsb
expe_1 = OrderedDict((
    ('data_type', 'networks'),
    ('corpus', '*'),
    #('debug' , 'debug101010') ,
    ('debug' , 'debug111111') ,
    ('model' , 'immsb')   ,
    ('K'     , '*')         ,
    ('hyper' , 'auto')     ,
    ('homo'  , 0) ,
    ('N'     , 'all')     ,
    ('repeat', '*'),
    ('measure', 8),
    ))
expe_1.update(expe_args)

# Hook
if expe_1['model'] == 'ibp':
    expe_1.update(hyper='fix')

assert(expe_1.keys()[:len(map_parameters)] == map_parameters.keys())

###################################
### Extract Resulst *** in: setting - out: table

### Make the ptx index
ptx = make_tensor_expe_index(expe_1, map_parameters)

### Output
## Forest setting
#print 'Forest:'
#print tabulate(map_parameters, headers="keys")
#finished =  1.0* rez.size - np.isnan(rez).sum()
#print '%.3f%% results over forest experimentations' % (finished / rez.size)

## Expe setting
#ptx = np.index_exp[0, :, 0, 0, 0, 1, 0, :]
print 'Expe 1:'
print tabulate([expe_1.keys(), expe_1.values()])
# Headers
headers = list(map_parameters['K'])
h_mask = 'mask all' if '11' in expe_1['debug'] else 'mask sub1'
h = expe_1['model'].upper() + ' / ' + h_mask
headers.insert(0, h)
# Row
keys = map_parameters['corpus']
keys = [''.join(k) for k in zip(keys, [' b/h', ' b/-h', ' -b/-h', ' -b/h'])]
## Results
table = rez[ptx]

try:
    table = np.column_stack((keys, table))
except ValueError, e:
    lgg.warn('ValueError, assumming repeat mean variance reduction: %d repetition' % table.shape[2])
    print table.shape
    #table_mean = np.char.array(table.mean(2))
    #table_std = np.char.array(table.std(2))
    table_mean = np.char.array(np.around(table.mean(2), decimals=3)).astype("|S20")
    table_std = np.char.array(np.around(table.std(2), decimals=3)).astype("|S20")
    table = table_mean + ' p2m ' + table_std
    table = np.column_stack((keys, table))

tablefmt = 'latex' # 'latex'
print
print tabulate(table, headers=headers, tablefmt=tablefmt, floatfmt='.3f')
print '\t\t--> precision'



