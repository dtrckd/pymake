#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

from frontend.manager import ModelManager
from frontend.frontendnetwork import frontendNetwork
from util.utils import *
from plot import *
from expe.spec import _spec_; _spec = _spec_()
from util.argparser import argparser
from util.math import reorder_mat

""" Inspect data on disk, for checking
    or updating results

    params
    ------
    zipf       : adjacency matrix + global degree
    burstiness : global + local + feature burstiness
    homo       : homophily based analysis
"""

### Config
config = dict(
    block_plot = True,
    save_plot = False,
    do           = 'zipf', # homo/zipf/burstiness/pvalue
    clusters_org = 'source' # source/model
)
config.update(argparser.generate(''))

### Specification
Corpuses = _spec.CORPUS_SYN_ICDM_1
#Corpuses = _spec.CORPUS_REAL_ICDM_1

Corpuses = _spec.CORPUS_NET_ALL

#Model = _spec.MODEL_FOR_CLUSTER_IBP
Model = _spec.MODEL_FOR_CLUSTER_IMMSB

### Simulation Output
if config.get('simulate'):
    print('''--- Simulation settings ---
    Build Corpuses %s''' % (str(Corpuses)))
    exit()

for corpus_pos, corpus_name in enumerate(Corpuses):
    frontend = frontendNetwork(config)
    data = frontend.load_data(corpus_name)
    data = frontend.sample()
    prop = frontend.get_data_prop()
    msg = frontend.template(prop)

    lgg.info('---')
    lgg.info(_spec.name(corpus_name))
    lgg.info('Expe: %s' % config['do'])
    lgg.info('---')

    if config['do'] == 'homo':
        ###################################
        ### Homophily Analysis
        ###################################

        print (corpus_name)
        homo_euclide_o_old, homo_euclide_e_old = frontend.homophily(sim='euclide_old')
        diff2 = homo_euclide_o_old - homo_euclide_e_old
        homo_euclide_o_abs, homo_euclide_e_abs = frontend.homophily(sim='euclide_abs')
        diff3 = homo_euclide_o_abs - homo_euclide_e_abs
        homo_euclide_o_dist, homo_euclide_e_dist = frontend.homophily(sim='euclide_dist')
        diff4 = homo_euclide_o_dist - homo_euclide_e_dist
        homo_comm_o, homo_comm_e = frontend.homophily(sim='comm')
        diff1 = homo_comm_o - homo_comm_e
        homo_text =  '''Similarity | Hobs | Hexp | diff\
                       \ncommunity   %.3f  %.3f %.3f\
                       \neuclide_old   %.3f  %.3f %.3f\
                       \neuclide_abs   %.3f  %.3f %.3f\
                       \neuclide_dist  %.3f  %.3f %.3f\
        ''' % ( homo_comm_o, homo_comm_e ,diff1,
               homo_euclide_o_old, homo_euclide_e_old, diff2,
               homo_euclide_o_abs, homo_euclide_e_abs,diff3,
               homo_euclide_o_dist, homo_euclide_e_dist, diff4)
        print (homo_text)

        #prop = frontend.get_data_prop()
        #print frontend.template(prop)

    elif config['do'] == 'zipf':
        ###################################
        ### Zipf Analisis
        ###################################

        ### Get the Class/Cluster and local degree information
        data_r = data
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
            msg =  'Getting Latent Classes from Latent Models %s' % Model['model']
            Model.update(corpus=corpus_name)
            model = ModelManager(config=config).load(Model)
            clusters = model.get_clusters(K, skip=1)
            #clusters = model.get_communities(K)
        except Exception as e:
            msg = 'Skypping reordering adjacency matrix: %s' % e

        ##############################
        ### Reordering Adjacency Mmatrix based on Clusters/Class/Communities
        ##############################
        if clusters is not None:
            print ('Reordering Adj matrix from `%s\':' % config.get('clusters_org'))
            print ('corpus: %s/%s, %s, Clusters size: %s' % (corpus_name, _spec.name(corpus_name), msg, K))
            data_r = reorder_mat(data, clusters)
        else:
            print( 'corpus: %s/%s, noo Reordering !' % (corpus_name, _spec.name(corpus_name)))
        print()

        ###################################
        ### Plotting
        ###################################

        ###################################
        ### Plot Adjacency matrix
        ###################################
        plt.figure()
        plt.suptitle(corpus_name)
        plt.subplot(1,2,1)
        adjshow(data_r, title='Adjacency Matrix', fig=False)
        #plt.figtext(.15, .1, homo_text, fontsize=12)

        ###################################
        ### Plot Degree
        ###################################
        plt.subplot(1,2,2)
        #plot_degree_(data, title='Overall Degree')
        plot_degree_poly(data)

        display(False)
    elif config['do'] == 'burstiness':
        ###################################
        ### Zipf Analisis (global burstiness) + local burstiness + feature burstiness
        ###################################

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
            msg =  'Getting Latent Classes from Latent Models %s' % Model['model']
            Model.update(corpus=corpus_name)
            model = ModelManager(config=config).load(Model)
            clusters = model.get_clusters(K, skip=1)
            #clusters = model.get_communities(K)
        except Exception as e:
            msg = 'Skypping reordering adjacency matrix: %s' % e


        # Global burstiness
        d, dc = degree_hist(adj_to_degree(data))
        gof = gofit(d, dc)
        fig = plt.figure()
        plot_degree(data, spec=True)

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
        #\#

        plt.plot(x, ylin , 'g--', label='power %.2f' % alpha)
        plt.legend()

        # Local burstiness
        comm = frontend.communities_analysis()
        clusters = comm['clusters']
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

        # Class burstiness
        plt.figure()
        hist, label = sorted_perm(comm['block_hist'], reverse=True)
        bins = len(hist)
        plt.bar(range(bins), hist)
        plt.xticks(np.arange(bins)+0.5, label)
        plt.xlabel('Class labels')
        plt.title('Blocks Size (max assignement)')

        display(config['block_plot'])

    elif config['do'] == 'pvalue':

        ####################
        ### Pvalue Table
        ####################

        d, dc = degree_hist(adj_to_degree(data))
        gof = gofit(d, dc)

        try:
            Table
        except NameError:
            Meas = [ 'pvalue', 'alpha', 'x_min', 'n_tail']; headers = Meas
            Table = np.empty((len(Corpuses), len(Meas)))
            Table = np.column_stack((_spec.name(Corpuses), Table))

        for i, v in enumerate(Meas):
            Table[corpus_pos, i+1] = gof[v]

###########################################
###########################################


### Table Format Printing
try:
    from tabulate import tabulate
    tablefmt = 'latex' # 'latex'
    print
    print (tabulate(Table, headers=headers, tablefmt=tablefmt, floatfmt='.3f'))

except NameError:
    pass

### Blocking Figures
if not config.get('save_plot'):
    display(True)
