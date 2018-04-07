import os
import numpy as np
from numpy import ma
from pymake import GramExp, ExpeFormat
from pymake.frontend.manager import ModelManager, FrontendManager
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
        self.model = ModelManager.from_expe(self.expe, load=True)

    @ExpeFormat.raw_plot('corpus')
    def __call__(self, frame,  attribute='_entropy'):
        ''' Plot figure group by :corpus:.
            Notes: likelihood/perplexity convergence report
        '''
        expe = self.expe

        data = self.load_some()
        if not data:
            self.log.warning('No data for expe : %s' % self.output_path)
            return

        values = data[attribute]
        values = self._to_masked(values)

        description = '/'.join((expe._refdir, os.path.basename(self.output_path)))

        ax = frame.ax()
        ax.plot(values, label=description, marker=frame.markers.next())
        ax.legend(loc='upper right',prop={'size':5})

    def _to_masked(self, lst, dtype=float):
        themask = lambda x:np.nan if x in ('--', 'None') else x
        if isinstance(lst, list):
            return ma.masked_invalid(np.array(list(map(themask, lst)), dtype=dtype))
        else:
            return themask(lst)


    @ExpeFormat.raw_plot
    def plot_unique(self, attribute='_entropy'):
        ''' Plot all figure in the same window.
            Notes: likelihood/perplexity convergence report '''
        expe = self.expe

        data = self.load_some()
        if not data:
            self.log.warning('No data for expe : %s' % self.output_path)
            return

        values = data[attribute]
        values = self._to_masked(values)

        description = '/'.join((expe._refdir, os.path.basename(self.output_path)))

        plt.plot(values, label=description, marker=_markers.next())
        plt.legend(loc='upper right',prop={'size':1})


    @ExpeFormat.plot()
    def fig(self, frame, attribute):
        ''' Plot all figure args is  `a:b..:c' (plot c by grouping by a, b...),
            if args is given, use for filename discrimination `key1[/key2]...'.
        '''
        expe = self.expe

        data = self.load_some()
        if not data:
            self.log.warning('No data for expe : %s' % self.output_path)
            return

        try:
            values = data[attribute]
        except KeyError as e:
            func_name = 'get_'+attribute.lstrip('_')
            if hasattr(self, func_name):
                values = getattr(self, func_name)(data)
            else:
                self.log.error('attribute unknown: %s' % attribute)
                return

        values = self._to_masked(values)

        burnin = 0
        description = self.get_description()

        ax = frame.ax()

        if expe.get('fig_xaxis'):
            xaxis = expe['fig_xaxis']
            if isinstance(xaxis, (tuple, list)):
                xaxis_name = xaxis[0]
                xaxis_surname = xaxis[1]
            else:
                xaxis_name = xaxis_surname = xaxis
            x = np.array(data[xaxis_name], dtype=int)
            xmax = frame.get('xmax',[])
            xmin = frame.get('xmin',[])
            xmax.append(max(x))
            xmax.append(min(x))
            frame.xmax = xmax
            frame.xmin = xmin
            frame.xaxis = xaxis_surname
        else:
            x = range(len(values))

        ax.plot(x[burnin:], values, label=description, marker=frame.markers.next())
        ax.legend(loc=expe.get('fig_legend',1), prop={'size':5})

        #if self.is_last_expe() and expe.get('fig_xaxis'):
        #    for frame in self.get_figs():
        #        xmax = max(frame.xmax)
        #        xmin = min(frame.xmin)
        #        xx = np.linspace(0,xmax, 10).astype(int)
        #        ax = frame.ax()
        #        ax.set_xticks(x)

        return


    @ExpeFormat.expe_repeat
    @ExpeFormat.table()
    def tab(self, array, floc, x, y, z):
        ''' Plot table according to parameter `x:y:z[-z2](param)'
            if args is given, use for filename discrimination `key1[/key2]...'
        '''
        expe = self.expe
        data = self.load_some()
        if not data:
            self.log.warning('No data for expe : %s' % self.output_path)

        if data and z in data and not expe.model.endswith('_gt'):
            # Extract from savec measure (.inf file).
            data = self._to_masked(max(data[z]))
            #data = self._to_masked(data[z][-1])
        else:
            # Compute it directly from the model.
            if not self.model:
                self.log.warning('No model for expe : %s' % self.output_path)
                return
            else:
                model = self.model

            if hasattr(model, 'compute_'+z.lstrip('_')):
                data = getattr(model, 'compute_'+z.lstrip('_'))(**expe)
            elif hasattr(self, 'get_'+z.lstrip('_')):
                data = getattr(self, 'get_'+z.lstrip('_'))()[-1]
            else:
                self.log.error('attribute unknown: %s' % z)
                return

        loc = floc(expe[x], expe[y], z)
        array[loc] = data
        return




    #
    # ml/model specific measure.
    #



    def get_perplexity(self, data):
        entropy = self._to_masked(data['_entropy'])
        nnz = self.model._data_test.shape[0]
        return 2 ** (-entropy / nnz)



if __name__ == '__main__':
    GramExp.generate().pymake(Plot)
