# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os.path

from random import choice
import itertools

import pymake.util.algo as A
from pymake.util.algo import gofit
from pymake.util.utils import *
from pymake.util.math import *
from .spec import _spec_; _spec = _spec_()

from pymake.plot import *
from pymake.plot import _markers, _colors

from tabulate import tabulate
from numpy import ma
import numpy as np
import scipy as sp
#np.set_printoptions(threshold='nan')
from sklearn.metrics import roc_curve, auc, precision_recall_curve



""" **kwargs is passed to the format function.
    The attributes curently in used in the globals dict are:
    * model_name (str)
    * corpus_name (str)
    * model (the model [ModelBase]
    * y (the data [Frontend])
    etc..
"""


def savefig_debug(**kwargs):
    # @Debug: does not make the variable accessible
    #in the current scope.
    globals().update(kwargs)
    path = '../results/networks/generate/'

    #################################################
    ### Plot Degree
    figsize=(3.8, 4.3)
    plt.figure(figsize=figsize)
    plot_degree_2_l(Y)
    plot_degree_poly(data, scatter=False)
    plt.title(title)

    #fn = path+fn+'_d_'+ globals()['K'] +'.pdf'
    fn = os.path.join(path, '%s_d_%s.pdf' % (fn, globals()['K']))
    print('saving %s' % fn)
    plt.savefig(fn, facecolor='white', edgecolor='black')

    return

def zipf(**kwargs):
    """ Local/Global Preferential attachment effect analysis """
    globals().update(kwargs)
    y = kwargs['y']
    N = y.shape[0]

    if Model['model'] == 'ibp':
        title = 'N=%s, K=%s alpha=%s, lambda:%s'% ( N, K, alpha, delta)
    elif Model['model'] == 'immsb':
        title = 'N=%s, K=%s alpha=%s, gamma=%s, lambda:%s'% (N, K, alpha, gmma, delta)
    elif Model['model'] == 'mmsb_cgs':
        title = 'N=%s, K=%s alpha=%s, lambda:%s'% ( N, K, alpha, delta)
    else:
        raise NotImplementedError

    ##############################
    ### Global degree
    ##############################
    #plot_degree_poly_l(Y)
    #plot_degree_poly(data, scatter=False)
    d, dc, yerr = random_degree(Y)
    god = gofit(d, dc)
    #plt.figure()
    #plot_degree_2((d,dc,yerr))
    #plt.title(title)
    plt.figure()
    plot_degree_2((d,dc,yerr), logscale=True)
    #plot_degree_2((d,dc,None), logscale=True)
    plt.title(title)

    if False:
        ### Just the gloabl degree.
        return

    print ('Computing Local Preferential attachment')
    ##############################
    ### Z assignement method
    ##############################
    limit_epoch = 50
    limit_class = 50
    now = Now()
    if model_name == 'immsb':
        ZZ = []
        #for _ in [Y[0]]:
        for _ in Y[:limit_epoch]: # Do not reflect real local degree !
            Z = np.empty((2,N,N))
            order = np.arange(N**2).reshape((N,N))
            if frontend.is_symmetric():
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
        ellapsed_time('Z formation', now)

    ##############################
    ### Plot all local degree
    ##############################
    plt.figure()
    # **max_assignement** evalutaion gives the degree concentration
    # for all clusters, when counting for
    # all the interaction for all other classess.
    # **modularity** counts only degree for interaction between two classes.
    # It appears that the modularity case concentration, correspond
    # to the interactions of concentration
    # of the maxèassignement case.
    #Clustering = ['modularity', 'max_assignement']
    clustering = 'modularity'
    comm = model.communities_analysis(data=Y[0], clustering=clustering)
    print( 'clustering method: %s, active clusters ratio: %f' % (clustering, len(comm['block_hist']>0)/float(theta.shape[1])))

    local_degree_c = {}
    ### Iterate over all classes couple
    if frontend.is_symmetric():
        #k_perm = np.unique( map(list, map(set, itertools.product(np.unique(clusters) , repeat=2))))
        k_perm =  np.unique(list(map(list, map(list, map(set, itertools.product(range(theta.shape[1]) , repeat=2))))))
    else:
        #k_perm = itertools.product(np.unique(clusters) , repeat=2)
        k_perm = itertools.product(range(theta.shape[1]) , repeat=2)
    for i, c in enumerate(k_perm):
        if i > limit_class:
            break
        if len(c) == 2:
            # Stochastic Equivalence (outer class)
            k, l = c
        else:
            # Comunnities (inner class)
            k = l = c.pop()

        degree_c = []
        YY = []
        if model_name == 'immsb':
            for y, z in zip(Y, ZZ): # take the len of ZZ if < Y
                y_c = np.zeros(y.shape)
                phi_c = np.zeros(y.shape)
                # UNDIRECTED !
                phi_c[(z[0] == k) & (z[1] == l)] = 1
                y_c = y * phi_c
                #degree_c += adj_to_degree(y_c).values()
                #yerr= None
                YY.append(y_c)
        elif model_name == 'ibp': # or Corpus !
            for y in Y:
                YY.append((y * np.outer(theta[:,k], theta[:,l] )).astype(int))

        ## remove ,old issue
        #if len(degree_c) == 0: continue
        #d, dc = degree_hist(degree_c)

        d, dc, yerr = random_degree(YY)
        if len(dc) == 0: continue
        #local_degree_c[str(k)+str(l)] = filter(lambda x: x != 0, degree_c)
        god =  gofit(d, dc)
        plot_degree_2((d,dc,yerr), logscale=True, colors=True, line=True)
    plt.title('Local Preferential attachment (Stochastic Block)')


    ##############################
    ### Blockmodel Analysis
    ##############################
    # Class Ties

    #plt.figure()
    ##local_degree = comm['local_degree']
    ##local_degree = local_degree_c # strong concentration on degree 1 !
    #label, hist = zip(*model.blockmodel_ties(Y[0]))
    #bins = len(hist)
    #plt.bar(range(bins), hist)
    #label_tick = lambda t : '-'.join(t)
    #plt.xticks(np.arange(bins)+0.5, map(label_tick, label))
    #plt.tick_params(labelsize=5)
    #plt.xlabel('Class Interactions')
    #plt.title('Weighted Harmonic mean of class interactions ties')


    if model_name == "immsb":

        # Class burstiness
        plt.figure()
        hist, label = clusters_hist(comm['clusters'])
        bins = len(hist)
        plt.bar(range(bins), hist)
        plt.xticks(np.arange(bins)+0.5, label)
        plt.xlabel('Class labels')
        plt.title('Blocks Size (max assignement)')
    elif model_name == "ibp":
        # Class burstiness
        plt.figure()
        hist, label = sorted_perm(comm['block_hist'], reverse=True)
        bins = len(hist)
        plt.bar(range(bins), hist)
        plt.xticks(np.arange(bins)+0.5, label)
        plt.xlabel('Class labels')
        plt.title('Blocks Size (max assignement)')

    #draw_graph_spring(y, clusters)
    #draw_graph_spectral(y, clusters)
    #draw_graph_circular(y, clusters)
    #adjshow(y, title='Adjacency Matrix')

    #adjblocks(y, clusters=comm['clusters'], title='Blockmodels of Adjacency matrix')
    #draw_blocks(comm)

    print ('density: %s' % (float(y.sum()) / (N**2)))


def pvalue(**kwargs):
    """ similar to zipf but compute pvalue and print table
        Parameters
        ==========
        type: pvalue type in (global, local, feature)
    """
    globals().update(kwargs)
    _type = kwargs.get('_type', 'local') # local | global
    lgg.info('using `%s\' burstiness' % _type)
    y = kwargs['y']
    N = y.shape[0]

    ## if model is None, work with dataset:
    #data = kwargs['data']
    #Y = [data]
    #K = max(frontend.clusters) +1
    #theta = np.zeros((data.shape[0], K))
    #theta[np.arange(data.shape[0]),  frontend.clusters] = 1
    ##\

    global Table
    Meas = [ 'pvalue', 'alpha', 'x_min', 'n_tail']; headers = Meas
    row_headers = _spec.name(Corpuses)

    if _type == 'global':
        Table = globals().get('Table', np.empty((len(row_headers), len(Meas), len(Y))))

        ### Global degree
        d, dc, yerr = random_degree(Y)
        for it_dat, data in enumerate(Y):
            d, dc = degree_hist(adj_to_degree(data))
            gof = gofit(d, dc)
            if not gof:
                continue

            for i, v in enumerate(Meas):
                Table[corpus_pos, i, it_dat] = gof[v]

    elif _type == 'local':
        ### Z assignement method
        now = Now()
        table_shape = (len(row_headers), len(Meas), K**2)
        Table = globals().get('Table', ma.array(np.empty(table_shape), mask=np.ones(table_shape)))

        if model_name == 'immsb':
            ZZ = []
            #for _ in [Y[0]]:
            for _ in Y[:5]: # Do not reflect real local degree !
                Z = np.empty((2,N,N))
                order = np.arange(N**2).reshape((N,N))
                if frontend.is_symmetric():
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
            ellapsed_time('Z formation', now)

        clustering = 'modularity'
        comm = model.communities_analysis(data=Y[0], clustering=clustering)
        print ('clustering method: %s, active clusters ratio: %f' % (clustering, len(comm['block_hist']>0)/float(theta.shape[1])))

        local_degree_c = {}
        ### Iterate over all classes couple
        if frontend.is_symmetric():
            #k_perm = np.unique( map(list, map(set, itertools.product(np.unique(clusters) , repeat=2))))
            k_perm =  np.unique(list(map(list, map(list, map(set, itertools.product(range(theta.shape[1]) , repeat=2))))))
        else:
            #k_perm = itertools.product(np.unique(clusters) , repeat=2)
            k_perm = itertools.product(range(theta.shape[1]) , repeat=2)
        for it_k, c in enumerate(k_perm):
            if it_k > 20:
                break
            if len(c) == 2:
                # Stochastic Equivalence (extra class bind
                k, l = c
                #continue
            else:
                # Comunnities (intra class bind)
                k = l = c.pop()

            degree_c = []
            YY = []
            if model_name == 'immsb':
                for y, z in zip(Y, ZZ): # take the len of ZZ if < Y
                    y_c = y.copy()
                    phi_c = np.zeros(y.shape)
                    # UNDIRECTED !
                    phi_c[(z[0] == k) & (z[1] == l)] = 1 #; phi_c[(z[0] == l) & (z[1] == k)] = 1
                    y_c[phi_c != 1] = 0
                    #degree_c += adj_to_degree(y_c).values()
                    #yerr= None
                    YY.append(y_c)
            elif model_name == 'ibp':
                for y in Y:
                    YY.append((y * np.outer(theta[:,k], theta[:,l])).astype(int))

            ## remove ,old issue
            #if len(degree_c) == 0: continue
            #d, dc = degree_hist(degree_c)

            d, dc, yerr = random_degree(YY)
            if len(dc) == 0: continue
            #local_degree_c[str(k)+str(l)] = filter(lambda x: x != 0, degree_c)
            gof =  gofit(d, dc)
            if not gof:
                continue

            for i, v in enumerate(Meas):
                Table[corpus_pos, i, it_k] = gof[v]

    elif _type == "feature":
        raise NotImplementedError
        ### Blockmodel Analysis
        if model_name == "immsb":
            # Class burstiness
            hist, label = clusters_hist(comm['clusters'])
            bins = len(hist)
        elif model_name == "ibp":
            # Class burstiness
            hist, label = sorted_perm(comm['block_hist'], reverse=True)
            bins = len(hist)
    else:
        raise NotImplementedError

    ### Table Format Printing
    if _end is True:
        # Function in (utils. ?)
        # Mean and standard deviation
        table_mean = np.char.array(np.around(Table.mean(2), decimals=3)).astype("|S20")
        table_std = np.char.array(np.around(Table.std(2), decimals=3)).astype("|S20")
        Table = table_mean + b' p2m ' + table_std

        # Table formatting
        Table = np.column_stack((row_headers, Table))
        tablefmt = 'latex' # 'latex'
        print()
        print (tabulate(Table, headers=headers, tablefmt=tablefmt, floatfmt='.3f'))


def homo(**kwargs):
    """ Hmophily test -- table output
        Parameters
        ==========
        type: similarity type in (natural, latent)
    """
    globals().update(kwargs)
    Y = kwargs['Y']
    _type = kwargs.get('_type', 'pearson') # contingency | pearson
    lgg.info('using `%s\' type' % _type)
    _sim = kwargs.get('_sim', 'latent') # natural | latent
    lgg.info('using `%s\' similarity' % _sim)
    force_table_print = False

    # class / self !
    global Table

    if _type == 'pearson':
        # No variance for link expecation !!!
        Y = [Y[0]]

        Meas = [ 'pearson coeff', '2-tailed pvalue' ]; headers = Meas
        row_headers = _spec.name(Corpuses)
        Table = globals().get('Table', np.empty((len(row_headers), len(Meas), len(Y))))

        ### Global degree
        d, dc, yerr = random_degree(Y)
        sim = model.similarity_matrix(sim=_sim)
        #plot(sim, title='Similarity', sort=True)
        #plot_degree(sim)
        for it_dat, data in enumerate(Y):

            #homo_object = data
            homo_object = model.likelihood()

            Table[corpus_pos, :,  it_dat] = sp.stats.pearsonr(homo_object.flatten(), sim.flatten())

    elif 'contingency':
        force_table_print = True
        Meas = [ 'esij', 'vsij' ]; headers = Meas
        row_headers = ['Non Links', 'Links']
        Table = globals().get('Table', np.empty((len(row_headers), len(Meas), len(Y))))

        ### Global degree
        d, dc, yerr = random_degree(Y)
        sim = model.similarity_matrix(sim=_sim)
        for it_dat, data in enumerate(Y):

            #homo_object = data
            homo_object = model.likelihood()

            Table[0, 0,  it_dat] = sim[data == 0].mean()
            Table[1, 0,  it_dat] = sim[data == 1].mean()
            Table[0, 1,  it_dat] = sim[data == 0].var()
            Table[1, 1,  it_dat] = sim[data == 1].var()


    ### Table Format Printing
    if _end is True or force_table_print is True:
        # Function in (utils. ?)
        # Mean and standard deviation
        table_mean = np.char.array(np.around(Table.mean(2), decimals=3)).astype("|S20")
        table_std = np.char.array(np.around(Table.std(2), decimals=3)).astype("|S20")
        Table = table_mean + b' p2m ' + table_std

        # Table formatting
        Table = np.column_stack((row_headers, Table))
        tablefmt = 'latex' # 'latex'
        print()
        print( tabulate(Table, headers=headers, tablefmt=tablefmt, floatfmt='.3f'))
        del Table



def roc_test(**kwargs):
    ''' AUC/ROC test report '''
    globals().update(kwargs)
    mask = model.get_mask()
    y_true, probas = model.mask_probas(data)
    fpr, tpr, thresholds = roc_curve(y_true, probas)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC %s (area = %0.2f)' % (_spec.name(model_name), roc_auc))

    #precision, recall, thresholds = precision_recall_curve( y_true, probas)
    #plt.plot(precision, recall, label='PR curve; %s' % (model_name ))

def perplexity(**kwargs):
    ''' likelihood/perplxity convergence report '''
    globals().update(kwargs)

    data = model.load_some()
    burnin = 5
    sep = ' '
    # Test perplexity not done for masked data. Usefull ?!
    #column = csv_row('likelihood_t')
    column = csv_row('likelihood')
    ll_y = [row.split(sep)[column] for row in data][5:]
    ll_y = np.ma.masked_invalid(np.array(ll_y, dtype='float'))
    plt.plot(ll_y, label=_spec.name(model_name))

_algo = 'Louvain'
_algo = 'Annealing'
def clustering(algo=_algo, **kwargs):
    globals().update(kwargs)

    mat = data
    #mat = phi

    alg = getattr(A, algo)(mat)
    clusters = alg.search()

    mat = draw_boundary(alg.hi_phi(), alg.B)
    #mat = draw_boundary(mat, clusters)

    adjshow(mat, algo)
    plt.colorbar()

def draw(**kwargs):
    y = kwargs['y']
    model = kwargs['model']

    clustering = 'modularity'
    comm = model.communities_analysis(data=y, clustering=clustering)

    clusters = comm['clusters']

    #y, l = reorder_mat(y, clusters, labels=True)
    #clusters = clusters[l]

    #adjblocks(y, clusters=clusters, title='Blockmodels of Adjacency matrix')
    #adjshow(reorder_mat(y, comm['clusters']), 'test reordering')

    draw_graph_circular(y, clusters)

    print (model.get_mask())



