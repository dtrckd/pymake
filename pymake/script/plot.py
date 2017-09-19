
#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
from numpy import ma
from pymake import ModelManager, FrontendManager, GramExp, ExpeFormat, ExpSpace
from pymake.plot import _markers, _colors, _linestyle

import matplotlib.pyplot as plt

import logging
lgg = logging.getLogger('root')


USAGE = """\
----------------
Plot utility :
----------------
"""


class Plot(ExpeFormat):

    def _preprocess(self):
        self.model = ModelManager.from_expe(self.expe)
        if self.model:
            # get _csv_type is any ...
            self.configure_model(self.model)

    @ExpeFormat.plot('corpus')
    def __call__(self, attribute='_entropy'):
        ''' likelihood/perplexity convergence report '''
        expe = self.expe

        data = self.load_some(self.output_path+'.inf')
        if not data:
            self.log.warning('No data for expe : %s' % self.output_path)
            return

        data = data[attribute]
        data = np.ma.masked_invalid(np.array(data, dtype='float'))

        burnin = 5
        description = '/'.join((expe._refdir, os.path.basename(self.output_path)))

        ax = self.gramexp.figs[expe.corpus].fig.gca()
        ax.plot(data, label=description, marker=_markers.next())
        ax.legend(loc='upper right',prop={'size':7})


    @ExpeFormat.plot
    def plot_unique(self, attribute='_entropy'):
        ''' likelihood/perplexity convergence report '''
        expe = self.expe

        data = self.load_some(self.output_path+'.inf')
        if not data:
            self.log.warning('No data for expe : %s' % self.output_path)
            return

        data = data[attribute]
        data = np.ma.masked_invalid(np.array(data, dtype='float'))

        burnin = 5
        description = '/'.join((expe._refdir, os.path.basename(self.output_path)))

        plt.plot(data, label=description, marker=_markers.next())
        plt.legend(loc='upper right',prop={'size':1})

if __name__ == '__main__':
    GramExp.generate().pymake(Plot)
