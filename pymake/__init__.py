# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

# __future__
try: basestring = basestring # python2
except NameError: basestring = (str, bytes) # python3

import  os
if os.environ.get('DISPLAY') is None:
    import matplotlib; matplotlib.use('Agg')


# This line is long (0.3 seconds !)
from pymake.expe.format import Corpus, Model, Script, ExpSpace, ExpVector, ExpTensor, ExpeFormat, ExpDesign
from pymake.expe.gramexp import GramExp

from pymake.frontend.frontend_io import SpecLoader
__spec = SpecLoader._default_spec()

from pymake.frontend.frontendtext import frontendText
from pymake.frontend.frontendnetwork import frontendNetwork
from pymake.frontend.manager import ModelManager, FrontendManager


#
# Erckelfault
#

#''' PRELOAD LIB '''
#import importlib
#_MODULES = ['community',
#            ('networkx', 'nx'),
#            ('numpy', 'np'),
#            ('scipy', 'sp'),
#            ('matplotlib.pyplot', 'plt')
#           ]
#
#for m in _MODULES:
#    try:
#        if type(m) is tuple:
#            mn = m[1]
#            m = m[0] if type(m) is tuple else m
#        else:
#            mn = m
#        globals()[mn] = importlib.import_module(m)
#    except ImportError:
#        print("* module `%s' unavailable" % (m))
#
