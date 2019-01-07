import numpy as np
from numpy import ma

from pymake import ExpeFormat
from pymake.manager import ModelManager, FrontendManager

import logging
lgg = logging.getLogger('root')

USAGE = """\
----------------
Inspect data on disk --or find the questions :
----------------
 |       or updating results
 |
 |   methods
 |   ------
 |   zipf       : adjacency matrix + global degree.
 |   burstiness : global + local + feature burstiness.
 |   homo       : homophily based analysis.
"""


from pymake.util.algo import gofit, Louvain, Annealing
from pymake.util.math import reorder_mat, sorted_perm
import matplotlib.pyplot as plt
from pymake.plot import plot_degree, degree_hist, adj_to_degree, plot_degree_poly, adjshow, plot_degree_2
from pymake.util.utils import colored

class ZipfNet(ExpeFormat):

    _default_expe = dict(
        _do           = ['zipf', 'source'],
        _data_type = 'networks'
    )

    def _preprocess(self):
        self.frontend = self.load_frontend()

    def init_fit_tables(self, _type, Y=[]):
        expe = self.expe
        if not hasattr(self.gramexp, 'tables'):
            corpuses = self.specname(self.gramexp.get_set('corpus'))
            models = self.gramexp.get_set('model')

            if not models:
                models = ['no_model']
            Meas = [ 'pvalue', 'alpha', 'x_min', 'n_tail']
            tables = {}
            for m in models:
                if _type == 'local':
                    table_shape = (len(corpuses), len(Meas), expe.K**2)
                    table = ma.array(np.empty(table_shape), mask=True)
                elif _type == 'global':
                    table = np.empty((len(corpuses), len(Meas), len(Y)))
                tables[m] = table
            self.gramexp.Meas = Meas
            self.gramexp.tables = tables
            Table = tables[expe.model]
        else:
            Table = self.gramexp.tables[expe.model]
            Meas = self.gramexp.Meas

        return Table, Meas

    @ExpeFormat.raw_plot
    def zipf(self, clusters_org='source'):
        ''' Zipf Analysis
            Local/Global Preferential attachment effect analysis

            Parameters
            ----------
            clusters_org: str
                cluster origin if from either ['source'|'model']
        '''
        expe = self.expe
        frontend = self.frontend

        #
        # Get the Class/Cluster and local degree information
        # Reordering Adjacency Mmatrix based on Clusters/Class/Communities
        #
        clusters = None
        K = None
        if clusters_org == 'source':
            clusters = frontend.get_clusters()
        elif clusters_org == 'model':
            model = ModelManager.from_expe(expe, load=True)
            #clusters = model.get_clusters(K, skip=1)
            #clusters = model.get_communities(K)
            try:
                clusters = Louvain.get_clusters(frontend.to_directed(), resolution=10)
            except:
                clusters = None
            if len(np.unique(clusters)) > 20 or False or True:
                self.log.info('Using Annealing clustering')
                clusters = Annealing(frontend.data, iterations=200, C_init=5, grow_rate=0).search()
            else:
                self.log.info('Using Louvain clustering')


        if clusters is None:
            lgg.error('No clusters here...passing')
            data_r = frontend.data
        else:
            block_hist = np.bincount(clusters)
            K = (block_hist != 0).sum()
            lgg.info('%d Clusters from `%s\':' % (K, clusters_org))
            data_r = reorder_mat(frontend.data, clusters)

        np.fill_diagonal(data_r, 0)

        from pymake.util.math import dilate
        dlt = lambda x : dilate(x) if x.sum()/x.shape[0]**2 < 0.1 else x
        ### Plot Adjacency matrix
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.tight_layout(pad=1.6)
        adjshow(dlt(data_r), title=self.specname(expe.corpus), ax=ax1)
        #plt.figtext(.15, .1, homo_text, fontsize=12)
        #plt.suptitle(self.specname(expe.corpus))

        ### Plot Degree
        plot_degree_poly(data_r, ax=ax2)

        ### Draw a more dilated adjancency matrix
        #dlt = lambda x : dilate(x, 4) if x.sum()/x.shape[0]**2 < 0.1 else x
        #plt.figure()
        #data_r += (np.random.random(data_r.shape) > 0.9999)
        #data_r = (data_r > 0).astype(int)
        #adjshow(dlt((data_r)))

        if expe._write:
            self.write_frames([fig], suffix='dd')

    @ExpeFormat.raw_plot
    def burstiness(self, clusters_org='source', _type='local'):
        '''Zipf Analisis
           (global burstiness) + local burstiness + feature burstiness
        '''
        expe = self.expe
        frontend = self.frontend
        data = frontend.data
        figs = []

        # Global burstiness
        d, dc = degree_hist(adj_to_degree(data), filter_zeros=True)
        fig = plt.figure()
        plot_degree(data, spec=True, title=self.specname(expe.corpus))
        #plot_degree_poly(data, spec=True, title=expe.corpus)

        gof = gofit(d, dc)
        if not gof:
            return

        alpha = gof['alpha']
        x_min = gof['x_min']
        y_max = gof['y_max']
        # plot linear law from power law estimation
        #plt.figure()
        idx = d.searchsorted(x_min)
        i = int(idx  - 0.1 * len(d))
        idx = i if i  >= 0 else idx
        x = d[idx:]
        ylin = np.exp(-alpha * np.log(x/float(x_min)) + np.log(y_max))
        #ylin = np.exp(-alpha * np.log(x/float(x_min)) + np.log((alpha-1)/x_min))

        # Hack xticks
        fig.canvas.draw() # !
        lim = plt.gca().get_xlim() # !
        locs, labels = plt.xticks()

        idx_xmin = locs.searchsorted(x_min)
        locs = np.insert(locs, idx_xmin, x_min)
        labels.insert(idx_xmin, plt.Text(text='x_min'))
        plt.xticks(locs, labels)
        plt.gca().set_xlim(lim)

        fit = np.polyfit(np.log(d), np.log(dc), deg=1)
        poly_fit = fit[0] *np.log(d) + fit[1]
        diff = np.abs(poly_fit[-1] - np.log(ylin[-1]))
        ylin = np.exp( np.log(ylin) + diff*0.75)
        #\#

        plt.plot(x, ylin , 'g--', label='power %.2f' % alpha)
        figs.append(plt.gcf())

        # Local burstiness

        #
        # Get the Class/Cluster and local degree information
        # Reordering Adjacency Mmatrix based on Clusters/Class/Communities
        #
        clusters = None
        K = None
        if clusters_org == 'source':
            clusters = frontend.get_clusters()
        elif clusters_org == 'model':
            model = ModelManager.from_expe(expe, load=True)
            #clusters = model.get_clusters(K, skip=1)
            #clusters = model.get_communities(K)
            clusters = Louvain.get_clusters(frontend.to_directed(), resolution=10)
            if len(np.unique(clusters)) > 20 or True:
                clusters = Annealing(frontend.data, iterations=200, C_init=5, grow_rate=0).search()

        if clusters is None:
            lgg.error('No clusters here...passing')
            return
        else:
            block_hist = np.bincount(clusters)
            K = (block_hist != 0).sum()
            lgg.info('%d Clusters from `%s\':' % (K, clusters_org))

        expe.K = K
        assert(not 'model' in expe)
        expe.model = 'no_model'
        #data_r, labels= reorder_mat(data, clusters, labels=True)
        Table,Meas = self.init_fit_tables(_type=_type)

        # Just inner degree
        f = plt.figure()
        ax = f.gca()
        #f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)

        # assume symmetric
        it_k = 0
        np.fill_diagonal(data, 0)
        for l in np.arange(K):
            for k in np.arange(K):
                if k != l:
                    continue

                ixgrid = np.ix_(clusters == k, clusters == l)

                if k == l:
                    title = 'Inner degree'
                    y = np.zeros(data.shape) # some zeros...
                    y[ixgrid] = data[ixgrid]
                    #ax = ax1
                else:
                    title = 'Outer degree'
                    y = np.zeros(data.shape) # some zeros...
                    y[ixgrid] = data[ixgrid]
                    #ax = ax2

                #
                title = ''
                #/#

                d, dc = degree_hist(adj_to_degree(y))
                if len(d) == 0: continue
                plot_degree_2((d,dc,None), logscale=True, colors=True, line=True, ax=ax, title=title)

                gof =  gofit(d, dc)
                if not gof:
                    continue

                for i, v in enumerate(Meas):
                    Table[self.corpus_pos, i, it_k] = gof[v] #* y.sum() / TOT
                it_k += 1

        plt.suptitle(self.specname(expe.corpus))
        figs.append(plt.gcf())

        # Features burstiness
        plt.figure()
        hist, label = sorted_perm(block_hist, reverse=True)
        bins = len(hist)
        plt.bar(range(bins), hist)
        plt.xticks(np.arange(bins)+0.5, label)
        plt.xlabel('Class labels')
        plt.title('Blocks Size (max assignement)')
        figs.append(plt.gcf())

        if expe._write:
            self.write_frames(figs)

        if self._it == self.expe_size -1:
            for _model, table in self.gramexp.tables.items():

                # Mean and standard deviation
                table_mean = np.char.array(np.around(table.mean(2), decimals=3)).astype("|S20")
                table_std = np.char.array(np.around(table.std(2), decimals=3)).astype("|S20")
                table = table_mean + b' $\pm$ ' + table_std

                # Table formatting
                corpuses = self.specname(self.gramexp.get_set('corpus'))
                table = np.column_stack((self.specname(corpuses), table))
                tablefmt = 'simple'
                table = self.tabulate(table, headers=['__'+_model.upper()+'__']+Meas, tablefmt=tablefmt, floatfmt='.3f')
                print()
                print(table)
                #if expe._write:
                #    base = '%s' % (clusters_org)
                #    self.write_frames(table, base=base)

    #@ExpeFormat.table
    def pvalue(self):
        ''' Compute Goodness of fit statistics '''
        expe = self.expe
        frontend = self.frontend
        data = frontend.data

        d, dc = degree_hist(adj_to_degree(data), filter_zeros=True)
        gof = gofit(d, dc)

        if not hasattr(self.gramexp, 'Table'):
            corpuses = self.specname(self.gramexp.get_set('corpus'))
            Meas = [ 'pvalue', 'alpha', 'x_min', 'n_tail']
            Table = np.empty((len(corpuses), len(Meas)))
            Table = np.column_stack((corpuses, Table))
            self.gramexp.Table = Table
            self.gramexp.Meas = Meas
        else:
            Table = self.gramexp.Table
            Meas = self.gramexp.Meas

        for i, v in enumerate(Meas):
            Table[self.corpus_pos, i+1] = gof[v]

        if self._it == self.expe_size -1:
            tablefmt = 'latex'
            print(colored('\nPvalue Table:', 'green'))
            print (self.tabulate(Table, headers=Meas, tablefmt=tablefmt, floatfmt='.3f'))


    def wsim(self):
        expe = self.expe
        frontend = self.frontend
        model = self.load_model(load=True)

        #y = model.generate(**expe)
        theta, phi = model.get_params()
        # NB mean/var
        mean, var = model.get_nb_ss()

        phi=mean

        print(theta.shape, theta.min(), theta.max())
        print(phi.shape, phi.min(), phi.max())

        adjshow(theta, title='theta', colorbar=True)
        adjshow(phi, title='Phi Mean', colorbar=True)

        y = self.frontend.adj()
        sim = np.random.poisson(theta.dot(phi).dot(theta.T))

        # l2 norm
        l2 = ((y - sim)**2).sum()**(0.5)
        print('l2', l2)

