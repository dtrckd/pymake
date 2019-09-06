#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import os
from multiprocessing import Process
from itertools import cycle

from local_utils import *
from lda_run import lda_gensim, preprocess_text

def display(block=False):
    #p = Process(target=_display)
    #p.start()
    plt.show(block=block)

def _display():
    os.setsid()
    plt.show()

def tag_from_csv(c):
    ## loglikelihood_Y, loglikelihood_Z, alpha, sigma, _K, Z_sum, ratio_MH_F, ratio_MH_W
    if c == 0:
        ylabel = 'mean eta'
        label = 'mean eta'
    elif c == 1:
        ylabel = 'var eta'
        label = 'var eta'
    elif c == 2:
        ylabel = 'mean alpha'
        label = 'mean alpha'
    elif c == 3:
        ylabel = 'var alpha'
        label = 'var alpha'
    elif c == 5:
        ylabel = 'perplexity'
        label = 'perplexity'

    return ylabel, label

def csv_row(s):
    #csv_typo = '# mean_eta, var_eta, mean_alpha, var_alpha, log_perplexity'
    if s == 'mean eta':
        row = 0
    elif s == 'var eta':
        row = 1
    elif s == 'mean alpha':
        row = 2
    elif s == 'var alpha':
        row = 3
    elif s == 'perplexity':
        row = 5
    else:
        row = s
    return row

def plot_csv(target_dir='', columns=0, sep=' ', separate=False, title=None):
    if type(columns) is not list:
        columns = [columns]

    title = title or 'LDA Inference'
    xlabel = 'Iterations'
    markers = cycle([ '+', '*', ',', 'o', '.', '1', 'p', ])
    if not separate:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel(xlabel)
        ax1.set_title(title)
    for column in columns:

        if separate:
            fig = plt.figure()
            plt.title(title)
            plt.xlabel('xlabel')
            ax1 = plt.gca()

        filen = os.path.join(os.path.dirname(__file__), "../PyNPB/data/", target_dir)
        with open(filen) as f:
            data = f.read()

        data = filter(None, data.split('\n'))
        data = [x.strip() for x in data if not x.startswith(('#', '%'))]

        ll_y = [row.split(sep)[column] for row in data]

        ylabel, label = tag_from_csv(column)
        ax1.set_ylabel(ylabel)

        #ax1.plot(ll_y, c='r',marker='x', label='log likelihood')
        ax1.plot(ll_y, marker=next(markers), label=label)
        leg = ax1.legend()
    plt.draw()

class ColorMap:
    def __init__(self, mat, cmap=None, pixelspervalue=20, minvalue=None, maxvalue=None, title='', ax=None):
        """ Make a colormap image of a matrix
        :key mat: the matrix to be used for the colormap.
        """
        if minvalue == None:
            minvalue = np.amin(mat)
            if maxvalue == None:
                maxvalue = np.amax(mat)
            if not cmap:
                cmap = plt.cm.hot
                if not ax:
                    #figsize = (np.array(mat.shape) / 100. * pixelspervalue)[::-1]
                    #self.fig = plt.figure(figsize=figsize)
                    #self.fig.set_size_inches(figsize)
                    #plt.axes([0, 0, 1, 1]) # Make the plot occupy the whole canvas
                    self.fig = plt.figure()
                    plt.axis('off')
                    plt.title(title)
                    implot = plt.imshow(mat, cmap=cmap, clim=(minvalue, maxvalue), interpolation='nearest')
                else:
                    ax.axis('off')
                    implot = ax.imshow(mat, cmap=cmap, clim=(minvalue, maxvalue), interpolation='nearest')
        def show(self):
            """ have the image popup """
            plt.show()
        def save(self, filename):
            """ save colormap to file"""
            plt.savefig(filename, fig=self.fig, facecolor='white', edgecolor='black')

def draw_adjmat(Y, title=''):
    plt.figure()
    plt.axis('off')
    plt.title('Adjacency matrix')
    plt.imshow(Y, cmap="Greys", interpolation="none", origin='upper')
    title = 'Adjacency matrix, N = %d\n%s' % (Y.shape[0], title)
    plt.title(title)

# Assume one mcmc file by directory
def plot_K_fix(sep=' ', columns=[0,'K'], target_dir='K_test'):
    bdir = os.path.join(os.path.dirname(__file__), "../../output", target_dir)

    # figure
    markers = cycle([ '+', '*', ',', 'o', '.', '1', 'p', ])
    fig = plt.figure()
    fig.canvas.set_window_title(target_dir)

    # for compared curves
    extra_c = []
    for i, column in enumerate(columns):

        # subplot
        ylabel, label = tag_from_csv(i)
        xlabel = 'iterations' if column == 0 else 'K'
        stitle = 'Likelihood convergence' if column == 0 else 'Likelihood comparaison'
        ax1 = fig.add_subplot(1, 2, i+1)
        plt.title(stitle)
        #ax1.set_title(stitle)
        ax1.set_xlabel(xlabel)
        if  column is 'K':
            support = np.arange(min(k_order),max(k_order)+1) # min max of K curve.
            k_order = sorted(range(len(k_order)), key=lambda k: k_order[k])
            extra_c = np.array(extra_c)[k_order]
            ax1.plot(support, extra_c, marker=next(markers))
            continue
        ax1.set_ylabel(ylabel)

        k_order = []
        # Assume one mcmc file by directory
        for dirname, dirnames, filenames in os.walk(bdir):
            if not 'mcmc' in filenames:
                continue

            _k = dirname.split('_')[-1]
            k_order.append(int(_k))
            filen = os.path.join(dirname, 'mcmc')
            with open(filen) as f:
                data = f.read()

            data = filter(None, data.split('\n'))
            data = [x.strip() for x in data if not x.startswith(('#', '%'))]
            curve = [row.split(sep)[column] for row in data]
            curve = np.ma.masked_invalid(np.array(curve, dtype='float'))
            extra_c.append(curve.max())
            ax1.plot(curve, marker=next(markers), label=_k)
            #leg = ax1.legend()

    plt.draw()

# several plot to see random consistancy
def plot_K_dyn(sep=' ', columns=[0,'K', 'K_hist'], target_dir='K_dyn'):
    bdir = os.path.join(os.path.dirname(__file__), "../../output", target_dir)

    # figure
    markers = cycle([ '+', '*', ',', 'o', '.', '1', 'p', ])
    fig = plt.figure()
    fig.canvas.set_window_title(target_dir)

    # for compared curves
    extra_c = []
    for i, column in enumerate(columns):

        # subplot
        ax1 = fig.add_subplot(2, 2, i+1)
        if column is 'K':
            plt.title('LL end point')
            ax1.set_xlabel('run')
            ax1.plot(extra_c, marker=next(markers))
            continue
        elif column is 'K_hist':
            plt.title('K distribution')
            ax1.set_xlabel('K')
            ax1.set_ylabel('P(K)')
            bins = int( len(set(k_order)) * 1)
            #k_order, _ = np.histogram(k_order, bins=bins, density=True)
            ax1.hist(k_order, bins, normed=True, range=(min(k_order), max(k_order)))
            continue
        else:
            ylabel, label = tag_from_csv(i)
            plt.title('Likelihood consistency')
            ax1.set_xlabel('iterations')
            ax1.set_ylabel(ylabel)

        k_order = []
        # Assume one mcmc file by directory
        for dirname, dirnames, filenames in os.walk(bdir):
            if not 'mcmc' in filenames:
                continue

            filen = os.path.join(dirname, 'mcmc')
            with open(filen) as f:
                data = f.read()

            data = filter(None, data.split('\n'))
            data = [x.strip() for x in data if not x.startswith(('#', '%'))]
            _k = data[csv_row('K')][-1]
            k_order.append(int(_k))
            curve = [row.split(sep)[column] for row in data]
            curve = np.ma.masked_invalid(np.array(curve, dtype='float'))
            extra_c.append(curve.max())
            ax1.plot(curve, marker=next(markers), label=_k)
            #leg = ax1.legend()

    plt.draw()

def plot_ibp(model, target_dir=None, block=False, columns=[0], separate=False, K=4):

    G = nx.from_numpy_matrix(model.Y(), nx.DiGraph())
    F = model.leftordered()
    W = model._W

    # Plot Adjacency Matrix
    draw_adjmat(model._Y)
    # Plot Log likelihood
    plot_csv(target_dir=target_dir, columns=columns, separate=separate)
    #W[np.where(np.logical_and(W>-1.6, W<1.6))] = 0
    #W[W <= -1.6]= -1
    #W[W >= 1.6] = 1

    # KMeans test
    clusters = kmeans(F, K=K)
    nodelist_kmeans = [k[0] for k in sorted(zip(range(len(clusters)), clusters), key=lambda k: k[1])]
    adj_mat_kmeans = nx.adjacency_matrix(G, nodelist=nodelist_kmeans).A
    draw_adjmat(adj_mat_kmeans, title='KMeans on feature matrix')
    # Adjacency matrix generation
    draw_adjmat(model.generate(nodelist_kmeans), title='Generated Y from ILFRM')

    # training Rescal
    R = rescal(model._Y, K)
    R = R[nodelist_kmeans, :][:, nodelist_kmeans]
    draw_adjmat(R, 'Rescal generated')

    # Networks Plots
    f = plt.figure()

    ax = f.add_subplot(121)
    title = 'Features matrix, K = %d' % model._K
    ax.set_title(title)
    ColorMap(F, pixelspervalue=5, title=title, ax=ax)

    ax = f.add_subplot(122)
    ax.set_title('W')
    img = ax.imshow(W, interpolation='None')
    plt.colorbar(img)

    f = plt.figure()
    ax = f.add_subplot(221)
    ax.set_title('Spectral')
    nx.draw_spectral(G, axes=ax)
    ax = f.add_subplot(222)
    ax.set_title('Spring')
    nx.draw(G, axes=ax)
    ax = f.add_subplot(223)
    ax.set_title('Random')
    nx.draw_random(G, axes=ax)
    ax = f.add_subplot(224)
    ax.set_title('graphviz')
    try:
        nx.draw_graphviz(G, axes=ax)
    except:
        pass

    display(block=block)

def newsgroups_class_distrib():
    from sklearn.datasets import fetch_20newsgroups
    ngroup_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=None)
    ngroup_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=None)
    test_data = ngroup_test.data
    train_data = ngroup_train.data
    test_groups = ngroup_test.target
    train_groups = ngroup_train.target

    n = 2000
    train_groups = train_groups[:n]
    test_groups = test_groups[:n]

    plt.figure()
    plt.hist(train_groups, 20, normed=True, range=(0, 19))
    plt.title("train groups")

    plt.figure()
    plt.hist(test_groups, 20, normed=True, range=(0, 19))
    plt.title("test groups")

    plt.show()

def plot_lda_comp (sep=' ', columns=['mean eta','var eta'], corpus=['20ngroups'], target_dir='', **conf):
    bdir = os.path.join(os.path.dirname(__file__), "../PyNPB/data/")
    #bdir = os.path.join(os.path.dirname(__file__), "../PyNPB/data/", target_dir)

    # figure
    markers = cycle([ '+', '*', ',', 'o', '.', '1', 'p', ])
    fig = plt.figure()
    fig.canvas.set_window_title(target_dir)

    # for compared curves
    for i, column in enumerate(columns):

        # subplot
        _column = csv_row(column)
        ylabel, label = tag_from_csv(_column)
        xlabel = 'iterations'
        stitle = label+ ' convergence'
        if len(columns) > 1:
            ax1 = fig.add_subplot(1, 2, i+1)
        else:
            ax1 = fig.add_subplot(1, 1, 1)
        plt.title(stitle)
        #ax1.set_title(stitle)
        ax1.set_xlabel(xlabel)
        if  column is 'K':
            support = np.arange(min(k_order),max(k_order)+1) # min max of K curve.
            k_order = sorted(range(len(k_order)), key=lambda k: k_order[k])
            extra_c = np.array(extra_c)[k_order]
            ax1.plot(support, extra_c, marker=next(markers))
            continue
        #ax1.set_ylabel(ylabel)

        # Assume one mcmc file by directory
        models = ['ldamodel', 'ldafullbaye']
        alpha = conf['alpha']
        Ks = conf['K_conv']
        #Ns =  [500, 1000, 2000, 3000, 11314]
        #Ns =  [500, 1000, 2000, 3000]
        Ns =  [2000, 5000]
        Ns =  conf['N_conv']
        for c in corpus:
            for K in Ks:
                for n in Ns:
                    for model in models:
                        label = '%s K=%s, %s' % (model, K, c)
                        fname = 'inference-%s_%s_%s_%s' % (model, K, alpha, n)
                        runs = map(str, conf['runs'])
                        curve_e = []
                        for run in runs:
                            if model == 'ldamodel' and conf['lookup_lda']:
                                filen = os.path.join(bdir,c, 'Fuzzy8', run,  fname)
                            else:
                                filen = os.path.join(bdir,c, target_dir, run,  fname)
                            with open(filen) as f:
                                data = f.read()

                            data = filter(None, data.split('\n'))
                            data = [x.strip() for x in data if not x.startswith(('#', '%'))]
                            curve = [row.split(sep)[_column] for row in data]
                            curve = np.ma.masked_invalid(np.array(curve, dtype='float'))
                            curve_e.append(curve)

                        curve_e = np.array(curve_e)
                        if column == 'perplexity':
                            curve_e = np.exp2(-curve_e)
                            #curve_e = -curve_e
                        elif column == 'mean eta' and model == 'ldafullbaye':
                            curve_e = 1/curve_e

                        #curve = curve_e.mean(axis=0)[1:]
                        #error = curve_e.std(axis=0)[1:]
                        curve = curve_e.mean(axis=0)[1:]
                        error = curve_e.std(axis=0)[1:]

                        #ax1.plot(curve, marker=next(markers), label=label)
                        _label = {'ldamodel': 'standard LDA', 'ldafullbaye':'conjugate LDA'}
                        ax1.errorbar(np.arange(len(curve)),  curve, yerr=error, marker=next(markers), label=_label[model])
                        leg = ax1.legend()

    plt.draw()

def plot_lda_end_point_kpc( target_dir='', **conf): # pass conf ...
    ############
    ### Load LDA
    save = False
    bdir = os.path.join(os.path.dirname(__file__), "../PyNPB/data/")
    alpha = conf['alpha']
    Ks = conf['Ks']
    models = ['ldamodel', 'ldafullbaye']
    runs = map(str, conf['runs'])

    corpus_ = [('20ngroups', 10000), ('nips12',123), ('reuter50', 4000)]
    shape = (len(corpus_), len(models),  len(Ks))
    perplexity_c = np.zeros(shape)
    perplexity_e = np.zeros(shape)

    for c_id, c_ in enumerate(corpus_):
        c = c_[0]
        n = c_[1]
        for model_id, model in enumerate(models):
            for k_id, K in enumerate(Ks):
                perplexity_m = []
                inference_time_m = []
                if model == 'ldamodel' and conf['lookup_lda']:
                    _bdir = os.path.join(bdir, c,'Fuzzy8')
                else:
                    _bdir = os.path.join(bdir,c,target_dir)
                for run in runs:
                    bdir_ = os.path.join(_bdir, run)
                    #lda = lda_gensim(load=True, K=K, bdir=bdir_, model=model, alpha=alpha, n=n)
                    try: lda = lda_gensim(load=True, K=K, bdir=bdir_, model=model, alpha=alpha, n=n)
                    except:
                        print 'file error in : %s, K=%s, alpha: %s, N=%s ; %s' % (model, K, alpha, n,bdir_ )
                        continue

                    fn = os.path.join(bdir_,  os.path.basename(lda.fname_i))
                    entropy = float(os.popen("awk '/./{line=$0} END{print line}' %s" % fn ).read().split()[-2])
                    perplexity_m.append(np.exp2(-entropy))

                ############
                ### Get Data
                #perplexity_c[model_id, k_id] = -perplexity
                perplexity_c[c_id, model_id, k_id] = np.mean(perplexity_m)
                perplexity_e[c_id, model_id, k_id] = np.std(perplexity_m)

    n_groups = len(Ks)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.1

    opacity = 0.9
    error_config = {'ecolor': '0.3'}
    colors = cycle([ 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w',])

    cpt = 0
    for c_id, c_ in enumerate(corpus_):
        c = c_[0]

        for m_id, model in enumerate(models):

            label = '%s, %s' % (c, model)
            plt.bar(index+(cpt)*bar_width, perplexity_c[c_id, m_id, :] , bar_width,
                    yerr=perplexity_e[c_id, m_id, :],
                    color=next(colors),
                    alpha=opacity,
                    error_kw=error_config,
                    label=label)
            cpt +=1

    plt.xlabel('#topic')
    plt.ylabel('Perplexity')
    plt.title('perplexity evolution over #topic')
    plt.xticks(index + bar_width, map(str, Ks))
    plt.legend(loc=1,prop={'size':6})

    plt.tight_layout()
    plt.show()

def plot_lda_end_point_pp( target_dir='', corpus=['20ngroups'], **conf): # pass conf ...
    ############
    ### Load LDA
    save = False
    bdir = os.path.join(os.path.dirname(__file__), "../PyNPB/data/")
    alpha = conf['alpha']
    Ks = conf['K_end']
    #Ks = [6,]
    models = ['ldamodel', 'ldafullbaye']
    Ns = conf['Ns']
    #Ks = [20] ; Ns = [1000]
    runs = map(str, conf['runs'])
    shape = (len(corpus), len(models),  len(Ks), len(Ns))
    perplexity_c = np.zeros(shape)
    inference_time_c = np.zeros(shape)
    perplexity_e = np.zeros(shape)
    inference_time_e = np.zeros(shape)

    for c_id, c in enumerate(corpus):
        for k_id, K in enumerate(Ks):
            #if c == 'reuter50':
            #    Ns = [ 500, 750, 1000, 1500, 2000, 2500, 3000, 3500,  4000]
            #else:
            #    Ns = [ 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
            for n_id, n in enumerate(Ns):
                for model_id, model in enumerate(models):
                    perplexity_m = []
                    inference_time_m = []
                    if model == 'ldamodel' and conf['lookup_lda']:
                        _bdir = os.path.join(bdir, c,'Fuzzy8')
                    else:
                        _bdir = os.path.join(bdir,c,target_dir)

                    for run in runs:
                        bdir_ = os.path.join(_bdir, run)
                        #lda = lda_gensim(load=True, K=K, bdir=bdir_, model=model, alpha=alpha, n=n)
                        try: lda = lda_gensim(load=True, K=K, bdir=bdir_, model=model, alpha=alpha, n=n)
                        except:
                            print 'file error in : %s, K=%s, alpha: %s, N=%s ; %s' % (model, K, alpha, n,bdir_ )
                            continue

                        fn = os.path.join(bdir_,  os.path.basename(lda.fname_i))
                        entropy = float(os.popen("awk '/./{line=$0} END{print line}' %s" % fn ).read().split()[-2])
                        perplexity_m.append(np.exp2(-entropy))
                        inference_time_m.append(lda.inference_time.total_seconds())

                    ############
                    ### Get Data
                    #perplexity_c[model_id, k_id, n_id] = -perplexity
                    perplexity_c[c_id, model_id, k_id, n_id] = np.mean(perplexity_m)
                    inference_time_c[c_id, model_id, k_id, n_id] = np.mean(inference_time_m)
                    perplexity_e[c_id, model_id, k_id, n_id] = np.std(perplexity_m)
                    inference_time_e[c_id, model_id, k_id, n_id] = np.std(inference_time_m)

    figs = {}
    for p in ['perplexity', 'time']:
        fig = plt.figure()
        fig.suptitle(p.title(), fontsize=14)
        figs[p] = fig

    x =  Ns
    for k_id, K in enumerate(Ks):
        # plt.figure() subplot with K=..., boring !
        fig3 = figs['perplexity']
        fig3 = fig3.add_subplot(len(Ks),1,k_id+1)
        fig3.set_title('K=%s' % K)
        fig4 = figs['time']
        fig4 = fig4.add_subplot(1,len(Ks),k_id+1)
        fig4.set_title('K=%s' % K)
        _label = {'ldamodel': 'standard LDA', 'ldafullbaye':'conjugate LDA'}
        for c_id, c in enumerate(corpus):
            if K == 20 and c == 'reuter50':
                continue
            elif K == 50 and c == '20ngroups':
                continue
            for model_id, model in enumerate(models):
                label = '%s, %s' % (_label[model], c)
                #fig3.errorbar(x, perplexity_c[c_id, model_id, k_id, :], yerr=perplexity_e[c_id, model_id, k_id, :], fmt='-*', label=label)

                fig4.errorbar(x, inference_time_c[c_id, model_id, k_id, :], yerr=inference_time_e[c_id, model_id, k_id, :], fmt='-*', label=label)

            ratio1 = perplexity_c[c_id, 1, k_id, :] / perplexity_c[c_id, 0, k_id, :]
            if c == '20ngroups':
                color = 'r'
                fig3.plot(x, ratio1, '-*', label='conjugate LDA / standard LDA, '+c , color=color)
                fig3.plot(x, [1]*len(x), '-' , color='g')
            elif c== 'reuter50':
                color = 'b'
                fig3.plot(x, ratio1, '-*', label='conjugate LDA / standard LDA, '+c , color=color)
                fig3.plot(x, [1]*len(x), '-' , color='g')

    for f in figs.values():
        ff = f.get_axes()
        for sf in ff:
            sf.set_xlabel('size of learning corpus')
            sf.legend(loc=4,prop={'size':10})

        if conf.get('save'):
            fn = f.texts[0].get_text() + '.eps'
            bdir = make_path('pyplot')
            f.savefig(os.path.join(bdir, fn))

    plt.draw()

def plot_lda_end_point( target_dir='', corpus=['20ngroups'], **conf): # pass conf ...
    ############
    ### Load LDA
    save = False
    corpus = corpus[0]
    bdir = os.path.join(os.path.dirname(__file__), "../PyNPB/data/", corpus)
    alpha = conf['alpha']
    Ks = conf['K_end']
    #Ks = [6,]
    models = ['ldamodel', 'ldafullbaye' ]
    Ns = conf['Ns']
    #Ks = [20] ; Ns = [1000]
    runs = map(str, conf['runs'])
    shape = (len(models),  len(Ks), len(Ns))
    purity_c = np.zeros(shape)
    precision_c = np.zeros(shape)
    perplexity_c = np.zeros(shape)
    inference_time_c = np.zeros(shape)
    purity_e = np.zeros(shape)
    precision_e = np.zeros(shape)
    perplexity_e = np.zeros(shape)
    inference_time_e = np.zeros(shape)

    from sklearn.datasets import fetch_20newsgroups
    ngroup_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=None)
    train_data = ngroup_train.data
    corpus, id2word = preprocess_text(train_data, bdir=bdir, corpus_name='train', load=True, save=False)
    train_classes = ngroup_train.target

    ngroup_test = ngroup_train
    true_classes = train_classes
    #ngroup_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=None)
    #test_data = ngroup_test.data
    #corpus_t, id2word_t = preprocess_text(test_data, bdir=bdir, corpus_name='test', load=True, save=False)
    #test_classes = ngroup_test.target

    for k_id, K in enumerate(Ks):
        for n_id, n in enumerate(Ns):
            for model_id, model in enumerate(models):
                perplexity_m = []
                precision_m = []
                purity_m = []
                inference_time_m = []
                if model == 'ldamodel' and conf['lookup_lda']:
                    _bdir = os.path.join(bdir, 'Fuzzy8')
                else:
                    _bdir = os.path.join(bdir,target_dir)
                for run in runs:
                    bdir_ = os.path.join(_bdir, run)
                    #lda = lda_gensim(load=True, K=K, bdir=bdir_, model=model, alpha=alpha, n=n)
                    try: lda = lda_gensim(load=True, K=K, bdir=bdir_, model=model, alpha=alpha, n=n)
                    except:
                        print 'file error in : %s, K=%s, alpha: %s, N=%s ; %s' % (model, K, alpha, n,bdir_ )
                        continue
                    print
                    print 'LDA prediction -- %s, K=%s, alpha: %s, N=%s ' % (model, K, alpha, n)

                    #################
                    ### Group Control
                    if K == 6 and len(ngroup_test.target_names) != 6:
                        # Wrap to subgroups
                        target_names = ['comp', 'misc', 'rec', 'sci', 'talk', 'soc']
                        map_ = dict([(0,5), (1,0), (2,0), (3,0), (4,0), (5,0), (6,1), (7,2), (8,2), (9,2), (10,2), (11,3), (12,3), (13,3), (14,3), (15,5), (16,4), (17,4), (18,4), (19,5)])
                        true_classes = set_v_to(true_classes, map_)
                    else:
                        target_names = ngroup_test.target_names
                    C = len(target_names)

                    ##############
                    ### Prediction
                    corpus_t = corpus
                    try:
                        confusion_mat = lda.confusion_matrix
                        map_kc = map_class2cluster_from_confusion(confusion_mat)
                    except:
                        predict_class = []
                        confusion_mat = np.zeros((K,C))
                        true_classes = train_classes
                        startt = datetime.now()
                        for i, d in enumerate(corpus_t):
                            d_t = lda.get_document_topics(d, minimum_probability=0.01)
                            t = max(d_t, key=lambda item:item[1])[0]
                            predict_class.append(t)
                            c = true_classes[i]
                            confusion_mat[t, c] += 1
                        last_d = ellapsed_time('LDA Prediction', startt)
                        predict_class = np.array(predict_class)
                        lda.confusion_matrix = confusion_mat

                        map_kc = map_class2cluster_from_confusion(confusion_mat)
                        #new_predict_class = set_v_to(predict_class, dict(map_kc))

                    purity = confusion_mat.max(axis=1).sum() / len(corpus_t)
                    print 'Purity (K=%s, C=%s, D=%s): %s' % (K, C, len(corpus_t), purity)

                    #precision = np.sum(new_predict_class == true_classes) / float(len(predict_class)) # equal !!!
                    precision = np.sum(confusion_mat[zip(*map_kc)]) / float(len(corpus_t))
                    print 'Ratio Groups Control: %s' % (precision)
                    if save:
                        lda.save(lda.fname)

                    fn = os.path.join(bdir_,  os.path.basename(lda.fname_i))
                    entropy = float(os.popen("awk '/./{line=$0} END{print line}' %s" % fn ).read().split()[-2])
                    perplexity_m.append(np.exp2(-entropy))
                    precision_m.append(precision)
                    purity_m.append(purity)
                    inference_time_m.append(lda.inference_time.total_seconds())

                ############
                ### Get Data
                purity_c[model_id, k_id, n_id] = np.mean(purity_m)
                precision_c[model_id, k_id, n_id] = np.mean(precision_m)
                #perplexity_c[model_id, k_id, n_id] = -perplexity
                perplexity_c[model_id, k_id, n_id] = np.mean(perplexity_m)
                inference_time_c[model_id, k_id, n_id] = np.mean(inference_time_m)
                purity_e[model_id, k_id, n_id] = np.std(purity_m)
                precision_e[model_id, k_id, n_id] = np.std(precision_m)
                perplexity_e[model_id, k_id, n_id] = np.std(perplexity_m)
                inference_time_e[model_id, k_id, n_id] = np.std(inference_time_m)

    figs = {}
    for p in ['purity', 'precision', 'perplexity', 'time']:
        fig = plt.figure()
        fig.suptitle(p.title(), fontsize=14)
        figs[p] = fig

    x =  Ns
    for k_id, K in enumerate(Ks):
        # plt.figure() subplot with K=..., boring !
        fig1 = figs['purity']
        fig1 = fig1.add_subplot(1,2,k_id+1)
        fig1.set_title(' K=%s' % K)
        fig2 = figs['precision']
        fig2 = fig2.add_subplot(1,2,k_id+1)
        fig2.set_title('K=%s' % K)
        fig3 = figs['perplexity']
        fig3 = fig3.add_subplot(1,2,k_id+1)
        fig3.set_title('K=%s' % K)
        fig4 = figs['time']
        fig4 = fig4.add_subplot(1,2,k_id+1)
        fig4.set_title('K=%s' % K)
        for model_id, model in enumerate(models):
            fig2.errorbar(x, precision_c[model_id, k_id, :], yerr=precision_e[model_id, k_id, :], fmt='-*', label=model)
            #fig1.errorbar(x, purity_c[model_id, k_id, :], yerr=purity_e[model_id, k_id, :], fmt='-*', label=model)
            #fig3.errorbar(x, perplexity_c[model_id, k_id, :], yerr=perplexity_e[model_id, k_id, :], fmt='-*', label=model)
            fig4.errorbar(x, inference_time_c[model_id, k_id, :], yerr=inference_time_e[model_id, k_id, :], fmt='-*', label=model)
        ratio1 = perplexity_c[1, k_id, :] / perplexity_c[0, k_id, :]
        fig3.plot(x, ratio1, '-*', label='conjugate LDA / sandard LDA' )
        fig3.plot(x, [1]*len(x), '-' )
        ratio2 = purity_c[1, k_id, :] / purity_c[0, k_id, :]
        fig1.plot(x, ratio2, '-*', label='conjugate LDA / standard LDA' )
        fig1.plot(x, [1]*len(x), '-' )

    for f in figs.values():
        ff = f.get_axes()
        for sf in ff:
            sf.set_xlabel('size of learning corpus')
            sf.legend(loc=1,prop={'size':10})

        if conf.get('save'):
            fn = f.texts[0].get_text() + '.eps'
            bdir = make_path('pyplot')
            f.savefig(os.path.join(bdir, fn))

    plt.draw()


def beta_test(hook_dir='', **conf):

    K = 6
    model = 'ldafullbaye'
    alpha = 'asymmetric'
    n = 2000


    bdir = './../PyNPB/data/20ngroups/Fuzzy29/1/'
    fn = 'inference_%s_%s_%s_%s' % (model, K, alpha, n)

    flda = lda_gensim(load=True, K=K, bdir=bdir, model=model, alpha=alpha, n=n)


    ####

    K = 6
    model = 'ldamodel'
    alpha = 'asymmetric'
    n = 2000


    bdir = './../PyNPB/data/20ngroups/Fuzzy29/1/'
    fn = 'inference_%s_%s_%s_%s' % (model, K, alpha, n)

    lda = lda_gensim(load=True, K=K, bdir=bdir, model=model, alpha=alpha, n=n)

    flda.print_topics()
    lda.print_topics()

    kl_comp = np.zeros((K, K))
    kl_div = sp.stats.entropy
    entropy = sp.stats.entropy
    beta_lda = lda.state.get_lambda()
    beta_flda = flda.state.get_lambda()
    # Nortmalize
    for topic in xrange(len(beta_lda)):
        beta_lda[topic] = beta_lda[topic] / beta_lda[topic].sum()
        beta_flda[topic] = beta_flda[topic] / beta_flda[topic].sum()

    # KL divergence
    for k_f in xrange(K):
        for k_l in xrange(K):
            KL = kl_div(beta_flda[k_f], beta_lda[k_l], base=2)
            kl_comp[k_f, k_l] = KL

    # Entropy
    entropy_l = []
    entropy_f = []
    for t in xrange(K):
        entropy_l.append(entropy(beta_lda[t], base=2) )
        entropy_f.append(entropy(beta_flda[t], base=2) )

    n_t2 = K* [[999, 999]]
    to_assign = range(K)
    cpt = 0

    #while len(to_assign) != 0 and cpt < 100:
    #    for t1 in np.random.permutation(range(K)):
    #        t1 = int(t1)
    #        KL = kl_comp[t1]
    #        #nt2 =  np.argmin(KL)
    #        for nt2, KL2 in np.random.permutation(zip(range(K), KL)):
    #            nt2 =  int(nt2)
    #            print n_t2
    #            if nt2 in map(lambda l: l[0], n_t2):
    #                if KL2 < n_t2[t1][1]:
    #                    n_t2[t1] = nt2, KL2
    #            else:
    #                n_t2[t1] = nt2, KL2
    #                to_assign.remove(nt2)

    #    cpt += 1

    #print cpt
    #print n_t2

    from local_utils import map_class2cluster_from_confusion
    confu =  map_class2cluster_from_confusion(kl_comp, minmax='min')
    entropy_r = []
    for i,j in confu:
        entropy_r.append( entropy_f[i] / entropy_l[j] )

    print entropy_r


    return

if __name__ ==  '__main__':
    block = True
    conf = argParse()

    hook_dir = 'Fuzzy7/'
    hook_dir = 'Fuzzy9/'
    hook_dir = 'Fuzzy8/'
    hook_dir = 'F22/' #eta + alpha vs auto
    hook_dir = 'F39/' # alpha vs auto
    hook_dir = 'Fuzzy29/'

    columns = ['mean eta', 'var eta']
    columns = ['perplexity' ]

    conf['lookup_lda'] = False
    #conf['runs'] = [1,2,3,4,5,6,7,8,9,10]
    #conf['runs'] = [1,5,6,]
    #conf['runs'] = [1,5,6,7,8,9]
    conf['runs'] = [1,2]

    conf['alpha'] = 'auto'
    conf['alpha'] = 'asymmetric'

    conf['N_conv'] = [2000]
    conf['K_conv'] = [6]
    conf['K_end'] = [6,20, 50]

    corpus = ['odp']; conf['N_conv'] = ['1000']; conf['Ns'] = [500, 1000,]# 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    corpus = ['nips12' ];conf['N_conv'] = ['123']; conf['Ns'] =  [50, 100, 123,]
    corpus = ['reuter50']; conf['Ns'] = [500, 1000, 2000, 3000, 4000, ]
    corpus = ['20ngroups']; conf['N_conv'] = ['2000'];
    conf['Ns'] = [ 500, 1000, 2000, 3000,  4000, 5000, 6000, 7000, 8000, 9000, 10000];
    #conf['Ns'] = [ 500, 750, 1000, 1500, 2000, 2500, 3000, 3500,  4000, 5000, 6000, 7000, 8000, 9000, 10000];

    #conf['runs'] = [3,4,5,6,7,]
    conf['runs'] = [1,2]

    for cc in columns:
        plot_lda_comp(columns=[cc],corpus=corpus, target_dir=hook_dir, **conf)

    corpus = ['reuter50', '20ngroups']
    #if 'nips12' not in corpus:
    #    plot_lda_end_point(target_dir=hook_dir, corpus=corpus,  **conf)
    plot_lda_end_point_pp(target_dir=hook_dir, corpus=corpus,  **conf)


    conf['runs'] = [1, 2]
    conf['Ks'] = [6, 10, 20, 30, 40, 50]
    #plot_lda_end_point_kpc( target_dir=hook_dir, **conf)
    #targets = ['20newsgroups/inference-ldafullbaye_symmetric_11314_10']
    #for t in targets:
    #    plot_csv(t, 0)
    #    plot_csv(t, 1)

    display(block)


