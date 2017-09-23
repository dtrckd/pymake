# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import  os
if os.environ.get('DISPLAY') is None:
    # plot in nil/void
    import matplotlib; matplotlib.use('Agg')
    print('==> Warning : Unable to load DISPLAY')
    print("To force a display try : `export DISPLAY=:0.0")



# This lines take a while
from pymake.expe.format import Spec, Corpus, Model, Script, ExpSpace, ExpVector, ExpTensor, ExpeFormat, ExpDesign, ExpGroup

from pymake.frontend.frontend_io import SpecLoader
#__spec = SpecLoader.get_atoms()

from pymake.expe.gramexp import GramExp
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
