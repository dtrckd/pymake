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

    # Documentation !
    _default_expe = dict(
        _label = lambda expe: '%s %s' % (expe._alias[expe.model], expe.delta) if expe.model in expe._alias else False,
        legend_size=10,
        _csv_sample = 2,
        fig_burnin = 0
    )

    def _preprocess(self):
        pass

    def _extract_data(self, z, data, *args):

        value = None

        if z in data:
            # Extract from saved measure (.inf file).
            if 'min' in args:
                value = self._to_masked(data[z]).min()
            else:
                value = self._to_masked(data[z]).max()
            #value = self._to_masked(data[z][-1])

        elif '@' in z:
            # Take the  argmax/argmin of first part to extract the second
            ag, vl = z.split('@')

            if 'min' in args:
                value = self._to_masked(data[ag]).argmin()
            else:
                value = self._to_masked(data[ag]).argmax()

            value = data[vl][value]

        else:
            # Compute it directly from the model.
            self.model = ModelManager.from_expe(self.expe, load=True)
            if not self.model:
                return
            else:
                model = self.model

            if hasattr(model, 'compute_'+z.lstrip('_')):
                value = getattr(model, 'compute_'+z.lstrip('_'))(**self.expe)
            elif hasattr(self, 'get_'+z.lstrip('_')):
                value = getattr(self, 'get_'+z.lstrip('_'))()[-1]
            else:
                self.log.error('attribute unknown: %s' % z)
                return

        return value


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

        if expe.get('_csv_sample'):
            s = int(expe['_csv_sample'])
            x = x[::s]
            values = values[::s]
            self.log.warning('Subsampling data: _csv_sample=%s' % s)

        if expe.get('_label'):
            label = expe['_label'](expe)
            description = label if label else description


        #fr = self.load_frontend()
        #E = fr.num_edges()
        #N = fr.num_nodes()
        #m = self._zeros_set_len
        #pop =

        if 'cumsum' in self.expe:
            values = np.cumsum(values)

        if 'fig_burnin' in self.expe:
            burnin = self.expe.fig_burnin
            x = x[burnin:]
            values = values[burnin:]


        ax.plot(x, values, label=description, marker=frame.markers.next())
        ax.legend(loc=expe.get('fig_legend',1), prop={'size':expe.get('legend_size',5)})

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
    def tab(self, array, floc, x, y, z, *args):
        ''' Plot table according to the syntax:
            * `x:y:Z [args1/args2]'

            Z value syntax
            -------------
            The third value of the triplet is the value to print in the table. There is sepcial syntac options:
            * If several value are seperate by a '-', then for each one a table will be print out.
            * is Z is of the forme Z= a@b. Then the value of the tab is of the form data[a][data[b].argmax()].
                It takes the value b according to the max value of a.

            Special Axis
            ------------
            is x or y can have the special keywords reserved values
            * _spec: each column of the table will be attached to each different expSpace in the grobal ExpTensorsV2.

            args* syntax
            ------------
            If args is given, it'll be used for filename discrimination `key1[/key2]...'
            Args can contain special keyworkd:
            * tex: will format the table in latex
            * max/min: change the defualt value to take in the .inf file (default is max)
            * rmax/rmin: is repeat is given, it will take the min/max value of the repeats (default is mean)
        '''
        expe = self.expe
        data = self.load_some()
        if not data:
            self.log.warning('No data for expe : %s' % self.output_path)
            data = {}

        value = self._extract_data(z, data)

        if value:
            loc = floc(expe[x], expe[y], z)
            array[loc] = value

        return




    #
    # ml/model specific measure.
    #


    def get_perplexity(self, data):
        entropy = self._to_masked(data['_entropy'])
        nnz = self.model._data_test.shape[0]
        return 2 ** (-entropy / nnz)

    #
    #
    # Specidic
    #

    def roc_evolution2(self, *args, _type='errorbar'):
        ''' AUC difference between two models against testset_ratio
            * _type : learnset/testset
            * _type2 : max/min/mean
            * _ratio : ration of the traning set to predict. If 100 _predictall will be true

        '''
        expe = self.expe
        if self.is_first_expe():
            D = self.D
            axis = ['_repeat', 'training_ratio', 'corpus', 'model']
            z = ['_entropy@_roc']
            #z = ['_entropy@_wsim']
            D.array, D.floc = self.gramexp.get_array_loc_n(axis, z)
            D.z = z
            D.axis = axis

        data = self.load_some()
        if not data:
            self.log.warning('No data for expe : %s' % self.output_path)
            return

        array = self.D.array
        floc = self.D.floc
        z = self.D.z

        # Building tensor
        for _z in z:

            pos = floc(expe, _z)
            value = self._extract_data(_z, data, *args)
            value = float(self._to_masked(value))

            if value:

                #if 'wsbm_gt' in self.expe.model and '_wsim' in _z:
                #    value /= 1000

                array[pos] = value

        if self.is_last_expe():
            # Plotting

            #Meas = self.specname(self.get_expset('training_ratio'))
            #corpuses = self.specname(self.get_expset('corpus'))
            #models = self.specname(self.get_expset('model'))
            Meas = self.get_expset('training_ratio')
            corpuses = self.get_expset('corpus')
            models = self.get_expset('model')

            axe1 = self.D.axis.index('corpus')
            axe2 = self.D.axis.index('model')

            figs = {}

            for corpus in corpuses:

                self.markers.reset()
                self.colors.reset()
                self.linestyles.reset()
                fig = plt.figure()
                ax = fig.gca()

                jitter = np.linspace(-1, 1, len(Meas))

                for ii, model in enumerate(models):
                    idx1 = corpuses.index(corpus)
                    idx2 = models.index(model)
                    table = array[:, :, idx1, idx2]
                    _mean = table.mean(0)
                    _std = table.std(0)
                    xaxis = np.array(list(map(int, Meas))) #+ jitter[ii]
                    if _type == 'errorbar':
                        ls = self.linestyles.next()
                        _std[_std> 0.15] = 0.15
                        _std[_std< -0.15] = -0.15
                        eb = ax.errorbar(xaxis , _mean, yerr=_std,
                                         fmt=self.markers.next(), ls=ls,
                                         #errorevery=3,
                                         #c=self.colors.next(),
                                         label=self.specname(model))
                        eb[-1][0].set_linestyle(ls)
                    elif _type == 'boxplot':
                        for meu, meas in enumerate(Meas):
                            bplot = table[:, meu,]
                            w = 0.2
                            eps = 0.01
                            ax.boxplot(bplot,  widths=w,
                                       positions=[meu],
                                       #positions=[meas],
                                       #positions=[int(meas)+(meu+eps)*w],
                                       whis='range' )

                if _type == 'errorbar':
                    ax.legend(loc='lower right',prop={'size':8})
                    ymin = array.min()
                    ymin = 0.45
                    ax.set_ylim(ymin)
                else:
                    ax.set_xticklabels(Meas)
                    #ticks = list(map(int, Meas))
                    ticks = list(range(1, len(Meas)))
                    ax.set_xticks(ticks)

                ax.set_title(self.specname(corpus), fontsize=20)
                ax.set_xlabel('percentage of the training edges')
                ax.set_ylabel('AUC-ROC')
                figs[corpus] = {'fig':fig, 'base': self.D.z[0]+'_evo'}

            if expe._write:
                self.write_frames(figs)


if __name__ == '__main__':
    GramExp.generate().pymake(Plot)

