#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


# __remove__ -> INTERNAL !
#from joblib import Parallel, delayed
import sys, multiprocessing

from pymake import GramExp


''' A Command line controler of Pymake '''


if __name__ == '__main__':

    zymake = GramExp.zymake()
    zyvar = zymake._conf

    ### Makes OUT Files
    if zyvar['_do'] == 'cmd':
        lines = zymake.make_commandline()
    elif zyvar['_do'] == 'path':
        lines = zymake.make_path(ftype=zyvar.get('_ftype', 'pk'), status=zyvar.get('_status'))
    elif zyvar['_do'] == 'show':
        lines = zymake.simulate()
    elif zyvar['_do'] == 'exec':
        lines = zymake.execute()
    elif zyvar['_do'] == 'burn':
        # @todo; parallelize Pymake()
        raise NotImplementedError('What parallel strategy ?')
    elif zyvar['_do'] == 'list':
        if zyvar.get('do_list') is True:
            print (zymake.spectable())
        elif zyvar.get('do_list') == 'atom':
            print (zymake.atomtable())
        elif zyvar.get('do_list') == 'atom_topos':
            print (zymake.atomtable(_type='topos'))
        elif 'do_list' in zyvar and not zyvar['do_list'] :
            print(zymake.help_short())
            print ('list what %s ?' % zyvar.get('do_list'))
        exit()
    else:
        raise NotImplementedError('zymake options unknow : %s' % zyvar)

    if zyvar.get('simulate'):
        # same as show !
        zymake.simulate()

    if lines is None:
        # catch signal ?
        exit()

    if 'script' in zyvar:
        script = ' '.join(zyvar['script'])
        lines = [' '.join((script, l)) for l in lines]

    zymake.simulate(halt=False, file=sys.stderr)
    print('\n'.join(lines), file=sys.stdout)

