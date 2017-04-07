#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

from copy import copy

from pymake import *

from pymake.plot import plot_degree, degree_hist, adj_to_degree, plot_degree_poly, adjshow
from pymake.util.algo import gofit
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

from pymake.util.utils import make_path
import os

_spec = GramExp.Spec()

#
# @todo: Merge this with plot do abstract
# a drawing class...
#
#
#
#
from pymake.frontend.frontend_io import _FIGS_PATH

def _name(n):
    return _spec.name(n).lower().replace(' ', '')

def full_path(fn):
    path = os.path.join(_FIGS_PATH, _name(fn))
    make_path(path)
    return path

def formatName(fun):
    def wrapper(*args, **kwargs):
        args = list(args)
        expe = copy(args[0])
        for k, v in expe.items():
            if isinstance(v, basestring):
                nn = _name(v)
            else:
                nn = v
            setattr(expe, k, nn)
        args[0] = expe
        f = fun(*args, **kwargs)
        return f
    return wrapper

@formatName
def write_figs(expe, figs, _suffix=None, _fn=None, ext='.png'):
    if type(figs) is list:
        _fn = '' if _fn is None else _name(_fn)+'_'
        for i, f in enumerate(figs):
            suffix = '_'+ _suffix if _suffix and len(figs)>1 else ''
            fn = ''.join([_fn, '%s_%s', ext]) % (expe.corpus, str(i) + suffix)
            print('Writings figs: %s' % fn)
            f.savefig(full_path(fn));
    elif issubclass(type(figs), dict):
        for c, f in figs.items():
            fn = ''.join([f.fn , ext])
            print('Writings figs: %s' % fn)
            f.fig.savefig(full_path(fn));
    else:
        print('ERROR : type of Figure unknow, passing')

def write_table(table, _fn=None, ext='.txt'):
    if isinstance(table, (list, np.ndarray)):
        _fn = '' if _fn is None else _name(_fn)+'_'
        fn = ''.join([_fn, 'table', ext])
        fn = full_path(fn)
        with open(fn, 'w') as _f:
            _f.write(table)
    elif isinstance(table, dict):
        for c, t in table.items():
            fn = ''.join([t.fn ,'_', _name(c), '_table', ext])
            fn = full_path(fn)
            with open(fn, 'w') as _f:
                _f.write(t.table)
    else:
        print('ERROR : type `%s\' of Table unknow, passing' % (type(table)))

#def debug():
#
#    plt.savefig(fn, facecolor='white', edgecolor='black')

