# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

# __future__
try: basestring = basestring # python2
except NameError: basestring = (str, bytes) # python3


''' BASETYPE '''
from collections import OrderedDict
class Expe(dict):
    pass
class ExpTensor(OrderedDict):
    pass

#import matplotlib; matplotlib.use('Agg')

from pymake.frontend.frontendtext import frontendText
from pymake.frontend.frontendnetwork import frontendNetwork
from pymake.frontend.manager import ModelManager, FrontendManager
from pymake.expe.format import ExpeFormat # GramExp+frontend_io

from pymake.util.argparser import GramExp


#
# Erckelfault
#

''' PRELOAD LIB '''
import importlib
_MODULES = ['community',
            ('networkx', 'nx'),
            ('numpy', 'np'),
            ('scipy', 'sp'),
            ('matplotlib.pyplot', 'plt')
           ]

for m in _MODULES:
    try:
        if type(m) is tuple:
            mn = m[1]
            m = m[0] if type(m) is tuple else m
        else:
            mn = m
        globals()[mn] = importlib.import_module(m)
    except ImportError:
        print("* module `%s' unavailable" % (m))

