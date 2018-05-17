import os
import numpy as np
import scipy as sp
from numpy import ma
from pymake import ExpTensor, GramExp, ExpeFormat, ExpSpace
from pymake.frontend.manager import ModelManager, FrontendManager



import itertools
from pymake.util.algo import gofit, Louvain, Annealing
from pymake.util.math import reorder_mat, sorted_perm, categorical, clusters_hist
from pymake.util.utils import Now, nowDiff, colored
import matplotlib.pyplot as plt
from pymake.plot import plot_degree, degree_hist, adj_to_degree, plot_degree_poly, adjshow, plot_degree_2, random_degree,  draw_graph_circular, draw_graph_spectral, draw_graph_spring, _markers
from pymake.core.format import tabulate

from sklearn.metrics import roc_curve, auc, precision_recall_curve



USAGE = '''\
----------------
Generate data  --or find the answers :
----------------
 |
 |   methods
 |   ------
 |   burstiness : global + local + feature burstiness.
 |   homo       : homophily based analysis.
 |
>> Examples
    parallel ./generate.py -k {}  ::: $(echo 5 10 15 20)
    generate --alpha 1 --gmma 1 -n 1000 --seed
'''

class GenNetwork(ExpeFormat):

    _default_expe = dict(
        block_plot = False,
        _write  = False,
        _do            = ['burstiness', 'global'], # default
        _mode         = 'generative',
        gen_size      = 1000,
        epoch         = 10 , # Gen,eration epoch
        limit_class   = 15, # Ignored ?
    )

    def _preprocess(self):
        expe = self.expe

        frontend = FrontendManager.load(expe)
        if frontend:
            self._N = frontend.getN()
            expe.symmetric = frontend.is_symmetric()
        else:
            self._N = expe.N
            expe.symmetric = True


        if expe._mode == 'predictive':
            ### Generate data from a fitted model
            model = ModelManager.from_expe(expe, load=True)

            try:
                # this try due to mthod modification entry in init not in picke object..
                expe.hyperparams = model.get_hyper()
            except Exception as e:
                self.log.warning('loading hyperparam error: %s' % e)
                if model is not None:
                    model._mean_w = 0
                    expe.hyperparams = 0

        elif expe._mode == 'generative':
            ### Generate data from a un-fitted model

            expe.alpha = 1
            expe.gmma = 1/2
            expe.delta = [0.5,0.5]

            if 'ilfm' in expe.model:
                keys_hyper = ('alpha','delta')
                hyper = (expe.alpha, expe.delta)
            else:
                keys_hyper = ('alpha','gmma','delta')
                hyper = (expe.alpha, expe.gmma, expe.delta)
            expe.hyperparams = dict(zip(keys_hyper, hyper))
            expe.hyper = 'fix' # dummy
            model = ModelManager.from_expe(expe, load=False)
            #model.update_hyper(hyper)

            ## Obsolete !
            #if 'ilfm' in expe.model:
            #    title = 'N=%s, K=%s alpha=%s, lambda:%s'% ( N, expe.K, expe.alpha, expe.delta)
            #elif 'immsb' in expe.model:
            #    title = 'N=%s, K=%s alpha=%s, gamma=%s, lambda:%s'% (N, expe.K, expe.alpha, expe.gmma, expe.delta)
            #elif 'mmsb' in expe.model:
            #    title = 'N=%s, K=%s alpha=%s, lambda:%s'% ( N, expe.K, expe.alpha, expe.delta)
            #else:
            #    raise NotImplementedError

            #expe.title = title

        else:
            raise NotImplementedError('What generation context ? predictive/generative..')

        self.log.info('=== GenNetworks === ')
        self.log.info('Mode: %s' % expe._mode)
        self.log.info('===')
        self.log.info('hyper: %s' % (str(expe.hyperparams)))

        self.frontend = frontend
        self.model = model

        if model is None:
            raise FileNotFoundError('No model for Expe at :  %s' % self.output_path)


        if expe._do[0] in ('burstiness','pvalue','homo', 'homo_mustach'):
            self._generate()


    def _generate(self):
        ''' Generate data. '''
        expe = self.expe
        model = self.model

        Y = []
        Theta = []
        Phi = []
        now = Now()
        for i in range(expe.epoch):
            y, theta, phi = model.generate(mode=expe._mode,
                                           N=self._N, K=expe.K,
                                           symmetric=expe.symmetric,
                                          )
            Y.append(y)
            Theta.append(theta)
            Phi.append(phi)
        self.log.info('Data Generation : %s second' % nowDiff(now))

        #if expe.symmetric:
        #    for y in Y:
        #        frontend.symmetrize(y)

        self._Y = Y
        self._Theta = Theta
        self._Phi = Phi


    def init_fit_tables(self, _type, Y=[]):
        expe = self.expe
        if not hasattr(self.gramexp, 'tables'):
            corpuses = self.specname(self.gramexp.get_set('corpus'))
            models = self.gramexp.get_set('model')
            Meas = [ 'pvalue', 'alpha', 'x_min', 'n_tail']
            tables = {}
            for m in models:
                if _type == 'local':
                    table_shape = (len(corpuses), len(Meas),2* expe.K**2)
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

    def init_roc_tables(self):
        expe = self.expe
        if not hasattr(self.gramexp, 'tables'):
            corpuses = self.specname(self.gramexp.get_set('corpus'))
            if not 'testset_ratio' in self.pt:
                Meas = ['20']
            else:
                Meas = self.gramexp.exp_tensor['testset_ratio']
            tables = ma.array(np.empty((len(corpuses), len(Meas),len(self.gramexp.getK('_repeat')), 2)), mask=True)
            self.gramexp.Meas = Meas
            self.gramexp.tables = tables
        else:
            tables = self.gramexp.tables
            Meas = self.gramexp.Meas

        return tables, Meas

    @ExpeFormat.raw_plot
    def burstiness(self, _type='all'):
        '''Zipf Analysis
           (global burstiness) + local burstiness + feature burstiness

           Parameters
           ----------
           _type : str
            type of burstiness to compute in ('global', 'local', 'feature', 'all')
        '''
        if self.model is None: return
        expe = self.expe
        figs = []

        Y = self._Y
        N = Y[0].shape[0]
        model = self.model

        if _type in ('global', 'all'):
            # Global burstiness
            d, dc, yerr = random_degree(Y)
            fig = plt.figure()
            title = 'global | %s, %s' % (self.specname(expe.get('corpus')), self.specname(expe.model))
            plot_degree_2((d,dc,yerr), logscale=True, title=title)

            figs.append(plt.gcf())

        if _type in  ('local', 'all'):
            # Local burstiness
            print ('Computing Local Preferential attachment')
            a, b = model.get_params()
            N,K = a.shape
            print('theta shape: %s'%(str((N,K))))
            now = Now()
            if 'mmsb' in expe.model:
                ### Z assignement method #
                ZZ = []
                for _i, _ in enumerate(Y):
                #for _ in Y: # Do not reflect real local degree !

                    theta = self._Theta[_i]
                    phi = self._Phi[_i]
                    Z = np.empty((2,N,N))
                    order = np.arange(N**2).reshape((N,N))
                    if expe.symmetric:
                        triu = np.triu_indices(N)
                        order = order[triu]
                    else:
                        order = order.flatten()
                    order = zip(*np.unravel_index(order, (N,N)))

                    for i,j in order:
                        Z[0, i,j] = categorical(theta[i])
                        Z[1, i,j] = categorical(theta[j])
                    Z[0] = np.triu(Z[0]) + np.triu(Z[0], 1).T
                    Z[1] = np.triu(Z[1]) + np.triu(Z[1], 1).T
                    ZZ.append( Z )
                self.log.info('Z formation %s second' % nowDiff(now))

            clustering = 'modularity'
            comm = model.communities_analysis(data=Y[0], clustering=clustering)
            print('clustering method: %s, active clusters ratio: %f' % (
                clustering,
                len(comm['block_hist']>0)/K))

            local_degree_c = {}
            ### Iterate over all classes couple
            if expe.symmetric:
                #k_perm = np.unique( map(list, map(set, itertools.product(np.unique(clusters) , repeat=2))))
                k_perm =  np.unique(list(map(list, map(list, map(set, itertools.product(range(K) , repeat=2))))))
            else:
                #k_perm = itertools.product(np.unique(clusters) , repeat=2)
                k_perm = itertools.product(range(K) , repeat=2)

            fig = plt.figure()
            for i, c in enumerate(k_perm):
                if isinstance(c,(np.int64, np.float64)):
                    k = l = c
                elif len(c) == 2:
                    # Stochastic Equivalence (outer class)
                    k, l = c
                else:
                    # Comunnities (inner class)
                    k = l = c.pop()
                #if i > expe.limit_class:
                #   break
                if k != l:
                    continue

                degree_c = []
                YY = []
                if 'mmsb' in expe.model:
                    for y, z in zip(Y, ZZ): # take the len of ZZ if < Y
                        y_c = np.zeros(y.shape)
                        phi_c = np.zeros(y.shape)
                        # UNDIRECTED !
                        phi_c[(z[0] == k) & (z[1] == l)] = 1
                        y_c = y * phi_c
                        #degree_c += adj_to_degree(y_c).values()
                        #yerr= None
                        YY.append(y_c)
                elif 'ilfm' in  expe.model: # or Corpus !
                    for _i , y in enumerate(Y):
                        theta = self._Theta[_i]
                        if theta.shape[1] <= max(k,l):
                            print('warning: not all block converted.')
                            continue
                        YY.append((y * np.outer(theta[:,k], theta[:,l] )).astype(int))

                d, dc, yerr = random_degree(YY)
                if len(d) == 0: continue
                title = 'local | %s, %s' % (self.specname(expe.get('corpus')), self.specname(expe.model))
                plot_degree_2((d,dc,yerr), logscale=True, colors=True, line=True,
                             title=title)
            figs.append(plt.gcf())

        # Blockmodel Analysis
        #if _type in  ('feature', 'all'):
        #    plt.figure()
        #    if 'mmsb' in expe.model:
        #        # Feature burstiness
        #        hist, label = clusters_hist(comm['clusters'])
        #        bins = len(hist)
        #        plt.bar(range(bins), hist)
        #        plt.xticks(np.arange(bins)+0.5, label)
        #        plt.xlabel('Class labels')
        #        plt.title('Blocks Size (max assignement)')
        #    elif 'ilfm' in expe.model:
        #        # Feature burstiness
        #        hist, label = sorted_perm(comm['block_hist'], reverse=True)
        #        bins = len(hist)
        #        plt.bar(range(bins), hist)
        #        plt.xticks(np.arange(bins)+0.5, label)
        #        plt.xlabel('Class labels')
        #        plt.title('Blocks Size (max assignement)')

        #    figs.append(plt.gcf())

        if expe._write:
            if expe._mode == 'predictive':
                base = '%s_%s' % (self.specname(expe.corpus), self.specname(expe.model))
            else:
                base = '%s_%s' % ('MG', self.specname(expe.model))
            self.write_frames(figs, base=base)
            return



    # @redondancy with burstiness !
    # in @ExpFormat.table
    def pvalue(self, _type='global'):
        """ similar to zipf but compute pvalue and print table

            Parameters
            ==========
            _type: str in [global, local, feature]
        """
        if self.model is None: return
        expe = self.expe
        figs = []

        Y = self._Y
        N = Y[0].shape[0]
        model = self.model

        Table, Meas = self.init_fit_tables(_type, Y)

        self.log.info('using `%s\' burstiness' % _type)

        if _type  == 'global':
            ### Global degree
            for it_dat, data in enumerate(Y):
                d, dc = degree_hist(adj_to_degree(data), filter_zeros=True)
                gof = gofit(d, dc)
                if not gof:
                    continue

                for i, v in enumerate(Meas):
                    Table[self.corpus_pos, i, it_dat] = gof[v]

        elif _type  == 'local':
            ### Z assignement method
            a, b = model.get_params()
            N,K = a.shape
            print('theta shape: %s'%(str((N,K))))
            now = Now()
            if 'mmsb' in expe.model:
                ZZ = []
                for _i, _ in enumerate(Y):
                #for _ in Y: # Do not reflect real local degree !
                    theta = self._Theta[_i]
                    phi = self._Phi[_i]
                    Z = np.empty((2,N,N))
                    order = np.arange(N**2).reshape((N,N))
                    if expe.symmetric:
                        triu = np.triu_indices(N)
                        order = order[triu]
                    else:
                        order = order.flatten()
                    order = zip(*np.unravel_index(order, (N,N)))

                    for i,j in order:
                        Z[0, i,j] = categorical(theta[i])
                        Z[1, i,j] = categorical(theta[j])
                    Z[0] = np.triu(Z[0]) + np.triu(Z[0], 1).T
                    Z[1] = np.triu(Z[1]) + np.triu(Z[1], 1).T
                    ZZ.append( Z )
                self.log.info('Z formation %s second', nowDiff(now))

            clustering = 'modularity'
            comm = model.communities_analysis(data=Y[0], clustering=clustering)
            print('clustering method: %s, active clusters ratio: %f' % (
                clustering,
                len(comm['block_hist']>0)/K))

            local_degree_c = {}
            ### Iterate over all classes couple
            if expe.symmetric:
                #k_perm = np.unique( map(list, map(set, itertools.product(np.unique(clusters) , repeat=2))))
                k_perm =  np.unique(list(map(list, map(list, map(set, itertools.product(range(K) , repeat=2))))))
            else:
                #k_perm = itertools.product(np.unique(clusters) , repeat=2)
                k_perm = itertools.product(range(K) , repeat=2)

            for it_k, c in enumerate(k_perm):
                if isinstance(c,(np.int64, np.float64)):
                    k = l = c
                elif len(c) == 2:
                    # Stochastic Equivalence (extra class bind
                    k, l = c
                    #continue
                else:
                    # Comunnities (intra class bind)
                    k = l = c.pop()
                #if i > expe.limit_class:
                #   break
                if k != l:
                    continue

                degree_c = []
                YY = []
                if 'mmsb' in expe.model:
                    for y, z in zip(Y, ZZ): # take the len of ZZ if < Y
                        y_c = y.copy()
                        phi_c = np.zeros(y.shape)
                        # UNDIRECTED !
                        phi_c[(z[0] == k) & (z[1] == l)] = 1 #; phi_c[(z[0] == l) & (z[1] == k)] = 1
                        y_c[phi_c != 1] = 0
                        #degree_c += adj_to_degree(y_c).values()
                        #yerr= None
                        YY.append(y_c)
                elif 'ilfm' in expe.model:
                    for _i , y in enumerate(Y):
                        theta = self._Theta[_i]
                        YY.append((y * np.outer(theta[:,k], theta[:,l])).astype(int))

                d, dc, yerr = random_degree(YY)
                if len(d) == 0: continue
                gof =  gofit(d, dc)
                if not gof:
                    continue

                for i, v in enumerate(Meas):
                    Table[self.corpus_pos, i, it_k] = gof[v]

        elif _type == 'feature':
            raise NotImplementedError

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
                table = tabulate(table, headers=['__'+_model.upper()+'__']+Meas, tablefmt=tablefmt, floatfmt='.3f')
                print()
                print(table)
                if expe._write:
                    if expe._mode == 'predictive':
                        base = '%s_%s_%s' % (self.specname(expe.corpus), self.specname(_model), _type)
                    else:
                        base = '%s_%s_%s' % ('MG', self.specname(_model), _type)
                    self.write_frames(table, base=base, ext='md')

    @ExpeFormat.raw_plot
    def draw(self):
        #if expe._mode == 'predictive':
        #    model = self.frontend
        #    y = model.data
        #    # move this in draw data
        #elif expe._mode == 'generative':
        #    model = self.model
        #    y = model.generate(**expe)
        model = self.model
        y = model.generate(**expe)

        #clustering = 'modularity'
        #print('@heeere, push commmunities annalysis outside static method of frontendnetwork')
        #comm = model.communities_analysis(y, clustering=clustering)
        #clusters = comm['clusters']
        #draw_graph_spring(y, clusters)
        #draw_graph_spectral(y, clusters)
        #draw_graph_circular(y, clusters)

        description = '/'.join((self.expe._refdir, os.path.basename(self.output_path)))
        adjshow(y, title=description)
        #adjblocks(y, clusters=comm['clusters'], title='Blockmodels of Adjacency matrix')
        #adjshow(reorder_mat(y, comm['clusters']), 'test reordering')
        #draw_blocks(comm)


    def homo(self, _type='pearson', _sim='latent'):
        """ Hmophily test -- table output
            Parameters
            ==========
            _type: similarity type in (contengency, pearson)
            _sim: similarity metric in (natural, latent)
        """
        if self.model is None: return
        expe = self.expe
        figs = []

        Y = self._Y
        N = Y[0].shape[0]
        model = self.model

        self.log.info('using `%s\' type' % _type)

        if not hasattr(self.gramexp, 'tables'):
            corpuses = self.specname(self.gramexp.get_set('corpus'))
            models = self.gramexp.get_set('model')
            tables = {}
            corpuses = self.specname(self.gramexp.get_set('corpus'))
            for m in models:
                if _type == 'pearson':
                    Meas = [ 'pearson coeff', '2-tailed pvalue' ]
                    table = np.empty((len(corpuses), len(Meas), len(Y)))
                elif _type == 'contingency':
                    Meas = [ 'natural', 'latent', 'natural', 'latent' ]
                    table = np.empty((2*len(corpuses), len(Meas), len(Y)))
                tables[m] = table

            self.gramexp.Meas = Meas
            self.gramexp.tables = tables
            table = tables[expe.model]
        else:
            table = self.gramexp.tables[expe.model]
            Meas = self.gramexp.Meas

        if _type == 'pearson':
            self.log.info('using `%s\' similarity' % _sim)
            # No variance for link expecation !!!
            Y = [Y[0]]

            ### Global degree
            d, dc, yerr = random_degree(Y)
            sim = model.similarity_matrix(sim=_sim)
            #plot(sim, title='Similarity', sort=True)
            #plot_degree(sim)
            for it_dat, data in enumerate(Y):
                #homo_object = data
                homo_object = model.likelihood()
                table[self.corpus_pos, :,  it_dat] = sp.stats.pearsonr(homo_object.flatten(), sim.flatten())

        elif _type == 'contingency':

            ### Global degree
            d, dc, yerr = random_degree(Y)
            sim_nat = model.similarity_matrix(sim='natural')
            sim_lat = model.similarity_matrix(sim='latent')
            step_tab = len(self.specname(self.gramexp.get_set('corpus')))
            for it_dat, data in enumerate(Y):

                #homo_object = data
                homo_object = model.likelihood()

                table[self.corpus_pos, 0,  it_dat] = sim_nat[data == 1].mean()
                table[self.corpus_pos, 1,  it_dat] = sim_lat[data == 1].mean()
                table[self.corpus_pos, 2,  it_dat] = sim_nat[data == 1].var()
                table[self.corpus_pos, 3,  it_dat] = sim_lat[data == 1].var()
                table[self.corpus_pos+step_tab, 0,  it_dat] = sim_nat[data == 0].mean()
                table[self.corpus_pos+step_tab, 1,  it_dat] = sim_lat[data == 0].mean()
                table[self.corpus_pos+step_tab, 2,  it_dat] = sim_nat[data == 0].var()
                table[self.corpus_pos+step_tab, 3,  it_dat] = sim_lat[data == 0].var()

        if self._it == self.expe_size -1:
            for _model, table in self.gramexp.tables.items():
                # Function in (utils. ?)
                # Mean and standard deviation
                table_mean = np.char.array(np.around(table.mean(2), decimals=3)).astype("|S20")
                table_std = np.char.array(np.around(table.std(2), decimals=3)).astype("|S20")
                table = table_mean + b' $\pm$ ' + table_std

                # Table formatting
                corpuses = self.specname(self.gramexp.get_set('corpus'))
                try:
                    table = np.column_stack((corpuses, table))
                except:
                    table = np.column_stack((corpuses*2, table))
                tablefmt = 'simple' # 'latex'
                table = tabulate(table, headers=['__'+_model.upper()+'__']+Meas, tablefmt=tablefmt, floatfmt='.3f')
                print()
                print(table)
                if expe._write:
                    base = '%s_homo_%s' % (self.specname(_model), _type)
                    self.write_frames(table, base=base, ext='md')

    @ExpeFormat.raw_plot('model')
    def homo_mustach(self, frame):
        """ Hmophily mustach
        """
        if self.model is None: return
        expe = self.expe
        figs = []

        Y = self._Y
        N = Y[0].shape[0]
        model = self.model

        if not hasattr(self.gramexp, 'tables'):
            corpuses = self.specname(self.gramexp.get_set('corpus'))
            models = self.gramexp.get_set('model')
            tables = {}
            corpuses = self.specname(self.gramexp.get_set('corpus'))
            for m in models:
                sim = [ 'natural', 'latent']
                Meas = ['links', 'non-links']
                table = {'natural': {'links':[], 'non-links':[]},'latent': {'links':[], 'non-links':[]} }
                tables[m] = table

            self.gramexp.Meas = Meas
            self.gramexp.tables = tables
            table = tables[expe.model]
        else:
            table = self.gramexp.tables[expe.model]
            Meas = self.gramexp.Meas


        ### Global degree
        d, dc, yerr = random_degree(Y)
        sim_nat = model.similarity_matrix(sim='natural')
        sim_lat = model.similarity_matrix(sim='latent')
        step_tab = len(self.specname(self.gramexp.get_set('corpus')))

        if not hasattr(self.gramexp._figs[expe.model], 'damax'):
            damax = -np.inf
        else:
            damax = self.gramexp._figs[expe.model].damax
        self.gramexp._figs[expe.model].damax = max(sim_nat.max(), sim_lat.max(), damax)
        for it_dat, data in enumerate(Y):

            #homo_object = data
            #homo_object = model.likelihood()

            table['natural']['links'].extend(sim_nat[data == 1].tolist())
            table['natural']['non-links'].extend(sim_nat[data == 0].tolist())
            table['latent']['links'].extend(sim_lat[data == 1].tolist())
            table['latent']['non-links'].extend(sim_lat[data == 0].tolist())


        if self._it == self.expe_size -1:
            for _model, table in self.gramexp.tables.items():
                ax = self.gramexp._figs[_model].fig.gca()

                bp = ax.boxplot([table['natural']['links']    ], widths=0.5,  positions = [1], whis='range')
                bp = ax.boxplot([table['natural']['non-links']], widths=0.5,  positions = [2], whis='range')
                bp = ax.boxplot([table['latent']['links']     ], widths=0.5,  positions = [4], whis='range')
                bp = ax.boxplot([table['latent']['non-links'] ], widths=0.5,  positions = [5], whis='range')

                ax.set_ylabel('Similarity')
                ax.set_xticks([1.5,4.5])
                ax.set_xticklabels(('natural', 'latent'), rotation=0)
                ax.set_xlim(0,6)

                nbox = 4
                top = self.gramexp._figs[_model].damax
                pos = [1,2,4,5]
                upperLabels = ['linked','    non-linked']*2
                #weights = ['light', 'ultralight']
                weights = ['normal', 'normal']
                for tick  in range(nbox):
                    ax.text(pos[tick], top+top*0.015 , upperLabels[tick],
                             horizontalalignment='center', weight=weights[tick%2])

                print(_model)
                t1 = sp.stats.ttest_ind(table['natural']['links'], table['natural']['non-links'])
                t2 = sp.stats.ttest_ind(table['latent']['links'], table['latent']['non-links'])
                print(t1)
                print(t2)

    @ExpeFormat.raw_plot('corpus', 'testset_ratio')
    def roc(self, frame,  _type='testset', _ratio=100):
        ''' AUC/ROC test report '''
        from sklearn.metrics import roc_curve, auc, precision_recall_curve
        expe = self.expe
        model = self.model
        data = self.frontend.data
        _ratio = int(_ratio)
        _predictall = (_ratio >= 100) or (_ratio < 0)
        if not hasattr(expe, 'testset_ratio'):
            setattr(expe, 'testset_ratio', 20)

        ax = frame.ax()

        if _type == 'testset':
            y_true, probas = model.mask_probas(data)
            if not _predictall:
                # take 20% of the size of the training set
                n_d =int( _ratio/100 * data.size * (1 - expe.testset_ratio/100) / (1 - _ratio/100))
                y_true = y_true[:n_d]
                probas = probas[:n_d]
            else:

                # Log Params statistics
                theta, phi = model.get_params()
                print('--- Params stats')
                print('Theta: shape: %s' %(str(theta.shape)))
                print('Theta: max: %s | min: %.4f | mean: %.4f | std: %.4f  ' % (theta.max(), theta.min(), theta.mean(), theta.std()))
                try:
                    print('Phi: shape: %s' %(str(phi.shape)))
                    print('Phi: max: %.4f | min: %.4f | mean: %.4f | std: %.4f  ' % (phi.max(), phi.min(), phi.mean(), phi.std()))
                except:
                    pass
                print('prediction:  links(1): %d | non-links(0): %d' % (y_true.sum(), (y_true==0).sum()))
                print('Prediction: probas stat: mean: %.4f | std: %.4f' % (probas.mean(), probas.std()))
                print('---')

        elif _type == 'learnset':
            n = int(data.size * _ratio)
            mask_index = np.unravel_index(np.random.permutation(data.size)[:n], data.shape)
            y_true = data[mask_index]
            probas = model.likelihood()[mask_index]

        try:
            fpr, tpr, thresholds = roc_curve(y_true, probas)
        except Exception as e:
            print(e)
            self.log.error('cant format expe : %s' % (self.output_path))
            return

        roc_auc = auc(fpr, tpr)
        description = '/'.join((expe._refdir, os.path.basename(self.output_path)))
        #description = self.specname(expe.model)
        ax.plot(fpr, tpr, label='ROC %s (area = %0.2f)' % (description, roc_auc), ls=frame.linestyle.next())
        plt.legend(loc='upper right',prop={'size':5})
        self.noplot = True

        #precision, recall, thresholds = precision_recall_curve( y_true, probas)
        #plt.plot(precision, recall, label='PR curve; %s' % (expe.model ))

        if self._it == self.expe_size -1:
            for c, f in self.gramexp._figs.items():
                ax = f.fig.gca()
                ax.plot([0, 1], [0, 1], linestyle='--', color='k', label='Luck')
                ax.legend(loc="lower right", prop={'size':5})

    def roc_evolution(self, _type='testset', _type2='max', _ratio=20, _type3='errorbar'):
        ''' AUC difference between two models against testset_ratio
            * _type : learnset/testset
            * _type2 : max/min/mean
            * _ratio : ration of the traning set to predict. If 100 _predictall will be true

        '''
        expe = self.expe
        model = self.model
        data = self.frontend.data
        _ratio = int(_ratio)
        _predictall = (_ratio >= 100) or (_ratio < 0)
        if not hasattr(expe, 'testset_ratio'):
            setattr(expe, 'testset_ratio', 20)
            self.testset_ratio_pos = 0
        else:
            self.testset_ratio_pos = self.pt['testset_ratio']

        table, Meas = self.init_roc_tables()

        #mask = model.get_mask()
        if _type == 'testset':
            y_true, probas = model.mask_probas(data)
            if not _predictall:
                # take 20% of the size of the training set
                n_d =int( _ratio/100 * data.size * (1 - expe.testset_ratio/100) / (1 - _ratio/100))
                y_true = y_true[:n_d]
                probas = probas[:n_d]
            else:
                pass

        elif _type == 'learnset':
            n = int(data.size * _ratio)
            mask_index = np.unravel_index(np.random.permutation(data.size)[:n], data.shape)
            y_true = data[mask_index]
            probas = model.likelihood()[mask_index]

        # Just the ONE:1
        #idx_1 = (y_true == 1)
        #idx_0 = (y_true == 0)
        #size_1 = idx_1.sum()
        #y_true = np.hstack((y_true[idx_1], y_true[idx_0][:size_1]))
        #probas = np.hstack((probas[idx_1], probas[idx_0][:size_1]))

        fpr, tpr, thresholds = roc_curve(y_true, probas)
        roc_auc = auc(fpr, tpr)

        table[self.corpus_pos, self.testset_ratio_pos, self.pt['_repeat'], self.model_pos] = roc_auc

        #precision, recall, thresholds = precision_recall_curve( y_true, probas)
        #plt.plot(precision, recall, label='PR curve; %s' % (expe.model ))

        if self._it == self.expe_size -1:

            # Reduce each repetitions
            take_type = getattr(np, _type2)
            t = ma.array(np.empty(table[:,:,0,:].shape), mask=True)
            t[:,:,0] = take_type(table[:, :, :, 0], -1)
            t[:,:,1] = take_type(table[:, :, :, 1], -1)
            table_mean = t.copy()
            t[:,:,0] = table[:, :, :, 0].std(-1)
            t[:,:,1] = table[:, :, :, 1].std(-1)
            table_std = t

            # Measure is comparaison of two AUC.
            id_mmsb = [i for i, s in enumerate(self.gramexp.exp_tensor['model']) if s.endswith('mmsb_cgs')][0]
            id_ibp = 1 if id_mmsb == 0 else 0
            table_mean = table_mean[:,:, id_mmsb] - table_mean[:,:, id_ibp]
            table_std = table_std[:,:, id_mmsb] + table_std[:,:, id_ibp]

            if _type2 != 'mean':
                table_std = [None] * len(table_std)

            fig = plt.figure()
            corpuses = self.specname(self.gramexp.get_set('corpus'))
            for i in range(len(corpuses)):
                if _type3 == 'errorbar':
                    plt.errorbar(list(map(int, Meas)), table_mean[i], yerr=table_std[i],
                                 fmt=_markers.next(),
                                 label=corpuses[i])
                elif _type3 == 'boxplot':
                    bplot = table[i,:,:,0] - table[i,:,:,1]
                    plt.boxplot(bplot.T, labels=corpuses[i])
                    fig.gca().set_xticklabels(Meas)

            plt.errorbar(Meas,[0]*len(Meas), linestyle='--', color='k')
            plt.legend(loc='lower left',prop={'size':7})

            # Table formatting
            #table = table_mean + b' $\pm$ ' + table_std
            table = table_mean

            corpuses = self.specname(self.gramexp.get_set('corpus'))
            table = np.column_stack((self.specname(corpuses), table))
            tablefmt = 'simple'
            headers = ['']+Meas
            table = tabulate(table, headers=headers, tablefmt=tablefmt, floatfmt='.3f')
            print()
            print(table)
            if expe._write:
                base = '%s_%s_%s' % ( _type, _type2, _ratio)
                figs = {'roc_evolution': ExpSpace({'fig':fig, 'table':table, 'base':base})}
                self.write_frames(figs)

    @ExpeFormat.raw_plot
    def clustering(self):
        algo = 'Louvain'
        algo = 'Annealing'
        data = self.frontend.data

        mat = data
        #mat = phi

        alg = getattr(A, algo)(mat)
        clusters = alg.search()

        mat = draw_boundary(alg.hi_phi(), alg.B)
        #mat = draw_boundary(mat, clusters)

        adjshow(mat, algo)
        plt.colorbar()





if __name__ == '__main__':

    GramExp.generate(usage=USAGE).pymake(GenNetwork)

