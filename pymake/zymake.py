#!/usr/bin/python -u
# -*- coding: utf-8 -*-

#from joblib import Parallel, delayed
import multiprocessing

from utils.argparser import argparser
from utils.utils import *
from frontend.frontend_io import *


USAGE = '''\
# Usage:
    zymake path[default] SPEC Filetype(pk|json|inf)
    zymake runcmd SPEC
'''

zyvar = argparser.zymake(USAGE)

### Makes OUT Files
if zyvar['OUT_TYPE'] == 'runcmd':
    source_files = make_forest_runcmd(zyvar['SPEC'])
elif zyvar['OUT_TYPE'] == 'path':
    source_files = make_forest_path(zyvar['SPEC'], zyvar['FTYPE'], status=zyvar['STATUS'])
else:
    raise NotImplementedError('zymake options unknow')


### Makes figures on remote / parallelize
#num_cores = int(multiprocessing.cpu_count() / 4)
#results_files = Parallel(n_jobs=num_cores)(delayed(expe_figures)(i) for i in source_files)
### ...and Retrieve the figure


print '\n'.join(source_files)

