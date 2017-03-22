
#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from numpy import ma
from pymake import ModelManager, FrontendManager, GramExp, ExpeFormat, ExpSpace

import logging
lgg = logging.getLogger('root')

USAGE = """\
----------------
Plot utility :
----------------
"""

from pymake.scripts.private import out
_spec = GramExp.Spec()

class Plot(ExpeFormat):

    def __init__(self, *args, **kwargs):
        super(Plot, self).__init__(*args, **kwargs)
        self.model = ModelManager.from_expe(self.expe)
        #self.frontend = FrontendManager.load(self.expe)

    @ExpeFormat.plot('corpus')
    def __call__(self, _type='likelihood'):
        ''' likelihood/perplexity convergence report '''
        expe = self.expe
        model = self.model

        ax = self.gramexp.figs[expe.corpus].fig.gca()

        data = model.load_some(get='likelihood')
        burnin = 5
        ll_y = np.ma.masked_invalid(np.array(data, dtype='float'))
        ax.plot(ll_y, label=_spec.name(expe.model))
        ax.legend(loc='upper right',prop={'size':7})


if __name__ == '__main__':
    GramExp.generate().pymake(Plot)
