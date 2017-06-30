#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
from pymake.expe.format import tabulate

from collections import OrderedDict

from pymake import ExpTensor
from frontend.frontend_io import *
from util.utils import *
from pymake.expe.gramexp import GramExp
from expe.spec import _spec

USAGE = '''\
# Usage:
    expe_meas [model] [K]
'''

#spec = _spec.EXPE_ICDM_R
exptensor = _spec['EXPE_ICDM_R_R']


###

#expe_args = GramExp.exp_tabulate(USAGE)
zymake = GramExp.exp_tabulate(exptensor, USAGE)

###################################################################
### Make Tensor Forest of results
rez = forest_tensor(zymake.make_path('json'), exptensor)

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
    ('K'     , 5)         ,
    ('hyper' , 'auto')     ,
    ('homo'  , 0) ,
    ('N'     , 'all')     ,
    ('repeat', '*'),
    ('measure', ':4'),
    ))
expe_1.update(expe_args)

# Hook
if expe_1['model'] == 'ibp':
    expe_1.update(hyper='fix')

assert(expe_1.keys()[:len(exptensor)] == exptensor.keys())

###################################
### Extract Resulst *** in: setting - out: table

### Make the ptx index
ptx = make_tensor_expe_index(expe_1, exptensor)

### Output
## Forest setting
#print 'Forest:'
#print tabulate(exptensor, headers="keys")
#finished =  1.0* rez.size - np.isnan(rez).sum()
#print '%.3f%% results over forest experimentations' % (finished / rez.size)

## Expe setting
#ptx = np.index_exp[0, :, 0, 0, 0, 1, 0, :]
print('Expe 1:')
print(tabulate([expe_1.keys(), expe_1.values()]))
# Headers
headers = [ 'global', 'precision', 'recall', 'K->']
h_mask = 'mask all' if '11' in expe_1['debug'] else 'mask sub1'
h = expe_1['model'].upper() + ' / ' + h_mask
headers.insert(0, h)
# Row
keys = exptensor['corpus']
#keys = [''.join(k) for k in zip(keys, [' b/h', ' b/-h', ' -b/-h', ' -b/h'])]
## Results
table = rez[ptx]

try:
    table = np.column_stack((keys, table))
except ValueError as e:
    hack_float = np.vectorize(lambda x : '{:.3f}'.format(float(x)))
    lgg.warn('ValueError, assumming repeat mean variance reduction: %d repetition' % table.shape[1])
    table_mean = np.char.array(hack_float(table.mean(1)), itemsize=100)
    table_std = np.char.array(hack_float(table.std(1)), itemsize=100)
    #table_mean = np.char.array(np.around(table.mean(1), decimals=3))
    #table_std = np.char.array(np.around(table.std(1), decimals=3))
    #table = table_mean + ' \pm ' + table_std
    table_mean[:, 3] = table_mean[:, 3] + ' p2m ' + table_std[:, 3]
    table = table_mean
    table = np.column_stack((keys, table))

tablefmt = 'latex' # 'latex'
print()
print(tabulate(table, headers=headers, tablefmt=tablefmt, floatfmt='.3f'))


