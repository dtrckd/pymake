#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


# __remove__ -> INTERNAL !
#from joblib import Parallel, delayed
import sys, multiprocessing

from pymake import GramExp
from frontend.manager import ModelManager, FrontendManager
from pymake.expe.spec import _spec


''' A Command line controler of Pymake '''


def Zymake(spec):
    commands = make_forest_conf(spec)
    if len(commands) == 1:
        frontend = FrontendManager.get(commands[0], load=True)
        model = ModelManager(commands[0])
        return frontend, model
    else:
        raise NotImplementedError('Multiple expe handle')

if __name__ == '__main__':

    zymake = GramExp.zymake()
    zyvar = zymake.expe

    ### Makes OUT Files
    if zyvar['_do'] == 'cmd':
        lines = zymake.make_commandline()
    elif zyvar['_do'] == 'path':
        lines = zymake.make_path(zyvar['_ftype'], status=zyvar['_status'])
    elif zyvar['_do'] == 'burn':
        server = 'hertog, macks, fuzzy, zombie-dust, victory, racer, tiger'
    elif zyvar['_do'] == 'show':
        zymake.simulate()
        exit()
    elif zyvar['_do'] == 'list':
        print (_spec.table())
        exit()
    else:
        raise NotImplementedError('zymake options unknow : %s' % zyvar)


    ### Makes figures on remote / parallelize
    #num_cores = int(multiprocessing.cpu_count() / 4)
    #results_files = Parallel(n_jobs=num_cores)(delayed(expe_figures)(i) for i in source_files)
    ### ...and Retrieve the figure

    if 'script' in zyvar:
        script = zyvar['script']
        lines = [' '.join((' '.join(script), l)) for l in lines]

    print('zymake request : %s\n  %s' %(zymake.expname(), zymake.exptable()), file=sys.stderr)
    print( '\n'.join(lines))

