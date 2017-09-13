
#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
from numpy import ma
from pymake import ModelManager, FrontendManager, GramExp, ExpeFormat, ExpSpace
import matplotlib.pyplot as plt

import logging
lgg = logging.getLogger('root')


USAGE = """\
----------------
Plot utility :
----------------
"""

from pymake.util import out
_spec = GramExp.Spec()

class Plot(ExpeFormat):

    def __init__(self, *args, **kwargs):
        super(Plot, self).__init__(*args, **kwargs)
        self.model = ModelManager.from_expe(self.expe)
        self.configure_model(self.model)
        #self.frontend = FrontendManager.load(self.expe)

    @ExpeFormat.plot('corpus')
    def __call__(self, attribute='_entropy'):
        ''' likelihood/perplexity convergence report '''
        expe = self.expe
        model = self.model

        ax = self.gramexp.figs[expe.corpus].fig.gca()

        data = self.load_some(self.output_path+'.inf')
        if not data:
            self.log.warning('No data for expe : %s' % self.output_path)
            return

        data = data[attribute]
        data = np.ma.masked_invalid(np.array(data, dtype='float'))

        #if type(y[0]) is list:
        #    y = [np.array(yy, dtype=float).mean() for yy in y]

        burnin = 5
        #description = _spec.name(expe.model)
        description = os.path.basename(self.output_path)
        ax.plot(data, label=description)
        ax.legend(loc='upper right',prop={'size':7})

    @ExpeFormat.plot
    def plot_unique(self, attribute='_entropy'):
        ''' likelihood/perplexity convergence report '''
        expe = self.expe
        model = self.model

        data = self.load_some(self.output_path+'.inf')[attribute]
        data = np.ma.masked_invalid(np.array(data, dtype='float'))

        burnin = 5
        plt.plot(data, label=_spec.name(expe.model))
        plt.legend(loc='upper right',prop={'size':7})

if __name__ == '__main__':
    GramExp.generate().pymake(Plot)
