#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from pymake import ExpTensor, ModelManager, FrontendManager, GramExp, ExpeFormat
from pymake.expe.spec import _spec

import logging
lgg = logging.getLogger('root')

USAGE = """\
----------------
Inspect data on disk, for questions :
----------------
 |       or updating results
 |
 |   methods
 |   ------
 |   zipf       : adjacency matrix + global degree.
 |   burstiness : global + local + feature burstiness.
 |   homo       : homophily based analysis.
     stats      : standard measure abd stats.
"""

Corpuses = _spec['CORPUS_SYN_ICDM']

#Model = _spec.MODEL_FOR_CLUSTER_IBP
#Model = _spec.MODEL_FOR_CLUSTER_IMMSB
Exp = ExpTensor ((
    ('corpus', Corpuses),
    ('data_type'    , 'networks'),
    ('refdir'        , 'debug111111') , # ign in gen
    #('model'        , 'mmsb_cgs')   ,
    ('model'        , 'immsb')   ,
    ('K'            , 10)        ,
    ('N'            , 'all')     , # ign in gen
    ('hyper'        , 'auto')    , # ign in gen
    ('homo'         , 0)         , # ign in gen
    ('repeat'      , '0')       ,
    #
    ('alpha', 1),
    ('gmma', 1),
    ('delta', [(1, 5)]),
))

class ExpeNetwork(ExpeFormat):

    @ExpeFormat.plot
    def zipf(self, **kwargs):
        ''' Zipf Analysis
            Local/Global Preferential attachment effect analysis
        '''
        expe = self.expe
        frontend = FrontendManager.load(expe)
        data_r = frontend.data

        ### Get the Class/Cluster and local degree information
        clusters = None
        K = None
        try:
            msg =  'Getting Cluster from Dataset.'
            clusters = frontend.get_clusters()
            if config.get('clusters_org') == 'model':
                if clusters is not None:
                    class_hist = np.bincount(clusters)
                    K = (class_hist != 0).sum()
                raise TypeError
        except TypeError:
            msg =  'Getting Latent Classes from Latent Models %s' % expe.model
            model = ModelManager.from_expe(expe)
            clusters = model.get_clusters(K, skip=1)
            #clusters = model.get_communities(K)
        except Exception as e:
            msg = 'Skypping reordering adjacency matrix: %s' % e

        ### Reordering Adjacency Mmatrix based on Clusters/Class/Communities
        if clusters is not None:
            print ('Reordering Adj matrix from `%s\':' % config.get('clusters_org'))
            print ('corpus: %s/%s, %s, Clusters size: %s' % (expe.corpus, _spec.name(expe.corpus), msg, K))
            data_r = reorder_mat(data_r, clusters)
        else:
            print( 'corpus: %s/%s, No Reordering !' % (expe.corpus, _spec.name(expe.corpus)))
        print()

        if expe.write:
            from private import out
            out.write_zipf(expe, data_r)
            return

        ### Plot Adjacency matrix
        plt.figure()
        plt.suptitle(_spec.name(expe.corpus))
        plt.subplot(1,2,1)
        adjshow(data_r, title='Adjacency Matrix', fig=False)
        #plt.figtext(.15, .1, homo_text, fontsize=12)

        ### Plot Degree
        plt.subplot(1,2,2)
        plot_degree_poly(data_r)

    @ExpeFormat.plot
    def burstiness(self, **kwargs):
        '''Zipf Analisis
           (global burstiness) + local burstiness + feature burstiness
        '''
        expe = self.expe
        frontend = FrontendManager.load(expe)
        data = frontend.data
        figs = []

        # Global burstiness
        d, dc = degree_hist(adj_to_degree(data))
        gof = gofit(d, dc)
        fig = plt.figure()
        plot_degree(data, spec=True, title=expe.corpus)
        #plot_degree_poly(data, spec=True, title=expe.corpus)

        alpha = gof['alpha']
        x_min = gof['x_min']
        y_max = gof['y_max']
        # plot linear law from power law estimation
        #plt.figure()
        idx = d.searchsorted(x_min)
        i = int(idx  - 0.1 * len(d))
        idx = i if i  >= 0 else idx
        x = d[idx:]
        ylin = np.exp(-alpha * np.log(x/float(x_min)) + np.log(y_max))
        #ylin = np.exp(-alpha * np.log(x/float(x_min)) + np.log((alpha-1)/x_min))

        # Hack xticks
        fig.canvas.draw() # !
        lim = plt.gca().get_xlim() # !
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

        # Local burstiness

        ### Get the Class/Cluster and local degree information
        try:
            msg =  'Getting Cluster from Dataset.'
            clusters = frontend.get_clusters()
            if config.get('clusters_org') == 'model':
                if clusters is not None:
                    class_hist = np.bincount(clusters)
                    K = (class_hist != 0).sum()
                raise TypeError
        except TypeError:
            msg =  'Getting Latent Classes from Latent Models %s' % expe.model
            model = ModelManager.from_expe(expe)
            clusters = model.get_clusters(K, skip=1)
            #clusters = model.get_communities(K)
        except Exception as e:
            msg = 'Skypping reordering adjacency matrix: %s' % e

        if clusters is None:
            lgg.error('No clusters here...passing')
            return

        comm = frontend.communities_analysis()
        K = len(comm['block_hist'])

        print ('clusters ground truth: %s' % ( comm['block_hist']))

        #data_r, labels= reorder_mat(data, clusters, labels=True)

        # Just inner degree
        #plt.figure()
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)

        # assume symmetric
        for l in np.arange(K):
            for k in np.arange(K):
                if k > l :
                    continue

                ixgrid = np.ix_(clusters == k, clusters == l)

                if k == l:
                    title = 'Inner degree'
                    #y = data[ixgrid]
                    y = np.zeros(data.shape) # some zeros...
                    y[ixgrid] = data[ixgrid]
                    ax = ax1
                else:
                    title = 'Outer degree'
                    y = np.zeros(data.shape) # some zeros...
                    y[ixgrid] = data[ixgrid]
                    ax = ax2

                d, dc = degree_hist(adj_to_degree(y))
                plot_degree_2((d,dc,None), logscale=True, colors=True, line=True, ax=ax, title=title)
        figs.append(plt.gcf())

        # Class burstiness
        plt.figure()
        hist, label = sorted_perm(comm['block_hist'], reverse=True)
        bins = len(hist)
        plt.bar(range(bins), hist)
        plt.xticks(np.arange(bins)+0.5, label)
        plt.xlabel('Class labels')
        plt.title('Blocks Size (max assignement)')
        figs.append(plt.gcf())

        if expe.write:
            from private import out
            out.write_figs(expe, figs)
            return

    # in @ExpFormat.table
    def pvalue(self, **kwargs):
        ''' Compute Goodness of fit statistics '''
        expe = self.expe
        frontend = FrontendManager.load(expe)
        data = frontend.data

        d, dc = degree_hist(adj_to_degree(data))
        gof = gofit(d, dc)

        if not hasattr(self.gramexp, 'Table'):
            Corpuses = list(map(_spec.name, self.gramexp.getCorpuses()))
            Meas = [ 'pvalue', 'alpha', 'x_min', 'n_tail']
            Table = np.empty((len(Corpuses), len(Meas)))
            Table = np.column_stack((Corpuses, Table))
            self.gramexp.Table = Table
            self.gramexp.Meas = Meas
        else:
            Table = self.gramexp.Table
            Meas = self.gramexp.Meas

        for i, v in enumerate(Meas):
            Table[self.corpus_pos, i+1] = gof[v]

        if self._it == self.expe_size -1:
            tablefmt = 'latex' # 'latex'
            print(colored('\nPvalue Table:', 'green'))
            print (tabulate(Table, headers=Meas, tablefmt=tablefmt, floatfmt='.3f'))

    # in @ExpFormat.table
    def stats(self, frontend='frontend'):
        ''' Show data stats '''
        expe = self.expe
        frontend = FrontendManager.load(expe)

        try:
            #@ugly debug
            Table = self.gramexp.Table
            Meas = self.gramexp.Meas
        except AttributeError:
            Corpuses = list(map(_spec.name, self.gramexp.getCorpuses()))
            Meas = [ 'nodes', 'edges', 'density']
            Table = np.empty((len(Corpuses), len(Meas)))
            Table = np.column_stack((Corpuses, Table))
            self.gramexp.Table = Table
            self.gramexp.Meas = Meas

        #print (frontend.get_data_prop())
        for i, v in enumerate(Meas):
            Table[self.corpus_pos, i+1] = getattr(frontend, v)()

        if self._it == self.expe_size -1:
            tablefmt = 'latex' # 'latex'
            print(colored('\nStats Table :', 'green'))
            print (tabulate(Table, headers=Meas, tablefmt=tablefmt, floatfmt='.3f'))

if __name__ == '__main__':
    from pymake.util.algo import gofit
    from pymake.util.math import reorder_mat, sorted_perm
    import matplotlib.pyplot as plt
    from pymake.plot import plot_degree, degree_hist, adj_to_degree, plot_degree_poly, adjshow, plot_degree_2, colored, tabulate

    config = dict(
        block_plot = False,
        write = False,
        _do           = 'zipf', # homo/zipf/burstiness/pvalue
        clusters_org = 'source', # source/model
        spec = Exp
    )

    GramExp.generate(config, USAGE).pymake(ExpeNetwork)
