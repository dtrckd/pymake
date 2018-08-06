import re, os, json

import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import write_dot

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as Colors

from pymake.util.utils import drop_zeros, nxG, Cycle
from pymake.util.math import adj_to_degree, clusters_hist, degree_hist, draw_square, random_degree, reorder_mat


from pymake.util.ascii_code import X11_colors, plt_colors
u_colors = Cycle(list(zip(*plt_colors))[1])
_markers = Cycle(['*', '+','x', 'o', '.', '1', 'p', '<', '>', 's'])
_colors = Cycle(['g','b','r','y','c','m','k'])
_linestyle = Cycle([ '--', ':', '-.'])
#_linestyle = Cycle(['solid' , 'dashed', 'dashdot', 'dotted'])

import logging
lgg = logging.getLogger('root')


def display(block=False):
    plt.show(block=block)

def _display():
    os.setsid()
    plt.show()

def tag_from_csv(c):
    ## loglikelihood_Y, loglikelihood_Z, alpha, sigma, _K, Z_sum, ratio_MH_F, ratio_MH_W
    if c == 0:
        ylabel = 'Iteration'
        label = 'Iteration'
    elif c in (1,2, 3):
        ylabel = 'loglikelihood'
        label = 'loglikelihood'

    return ylabel, label

# Obsolete
def csv_row(s):
    if s == 'Iteration':
        row = 0
    if s == 'Timeit':
        row = 1
    elif s in ('loglikelihood', 'likelihood', 'perplexity'):
        row = 2
    elif s in ('loglikelihood_t', 'likelihood_t', 'perplexity_t'):
        row = 3
    elif s == 'K':
        row = 4
    elif s == 'alpha':
        row = 5
    elif s == 'gamma':
        row = 6
    else:
        row = s
    return row

def surf(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, z)
    return ax

def plot(P, logscale=False, colors=False, line=True, ax=None, title=None, sort=False):
    """ General Plot
    Parameters
    ==========
    P: array or tuple
        y or (x, y), the curve (and optionnal, the x-axis)
    """
    if ax is None:
        # Note: difference betwwen ax and plt method are the get_ and set_ suffix
        ax = plt.gca()

    if isinstance(P, (list, tuple)):
        x, y = P
    else:
        y = P
        x = np.arange(len(y))

    ### Math
    if sort is True:
        x = sorted(y)

    ### Matplotlib
    if logscale:
        ax.set_xscale('log'); ax.set_yscale('log')
    if title:
        ax.set_title(title)

    c = next(_colors) if colors else 'b'
    m = next(_markers) if colors else 'o'
    l = '--' if line else None

    #ax.figure()
    ax.plot(x, y, c=c, marker=m, ls=l)


def log_binning(counter_dict,bin_count=35):
    max_x = np.log10(max(counter_dict.keys()))
    max_y = np.log10(max(counter_dict.values()))
    max_base = max([max_x,max_y])

    min_x = np.log10(min(drop_zeros(counter_dict.keys())))

    bins = np.logspace(min_x,max_base,num=bin_count)

    # Based off of: http://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
    #bin_means_y = (np.histogram(counter_dict.keys(),bins,weights=counter_dict.values())[0] / np.histogram(counter_dict.keys(),bins)[0])
    #bin_means_x = (np.histogram(counter_dict.keys(),bins,weights=counter_dict.keys())[0] / np.histogram(counter_dict.keys(),bins)[0])
    bin_means_y = np.histogram(counter_dict.keys(),bins,weights=counter_dict.values())[0]
    bin_means_x = np.histogram(counter_dict.keys(),bins,weights=counter_dict.keys())[0]
    return bin_means_x,bin_means_y


def plot_degree(y, spec=False,logscale=True, title=None, ax=None):
    """ Degree plot """
    if ax is None:
        # Note: difference betwwen ax and plt method are the get_ and set_ suffix
        ax = plt.gca()

    # To convert normalized degrees to raw degrees
    #ba_c = {k:int(v*(len(ba_g)-1)) for k,v in ba_c.iteritems()}
    if isinstance(y, np.ndarray) and y.ndim == 2:
        ba_c = adj_to_degree(y)
        d, dc = degree_hist(ba_c)
    else:
        # assume list
        d = np.arange(len(y))
        dc = sorted(y)

    ### Matplotlib
    if logscale:
        ax.set_xscale('log'); ax.set_yscale('log')
    if title:
        ax.set_title(title)

    ax.scatter(d,dc,c='b',marker='o')
    #plt.scatter(ba_x,ba_y,c='r',marker='s',s=50)

    if spec is True:
        ax.set_xlim(left=1)
        ax.set_ylim((.9,1e3))
        ax.set_xlabel('Degree')
        ax.set_ylabel('Counts')

def plor_degree_polygof(y, gof):
    # I dunno structore for plot ?
    pass

def plot_degree_poly(y, scatter=True, spec=True, title=None, ax=None, logscale=True):
    """ Degree plot along with a linear regression of the distribution.
    if scatter is false, draw only the linear approx"""
    # To convert normalized degrees to raw degrees
    #ba_c = {k:int(v*(len(ba_g)-1)) for k,v in ba_c.iteritems()}
    ba_c = adj_to_degree(y)
    d, dc = degree_hist(ba_c, filter_zeros=True)

    ### Matplotlib
    if ax is None:
        # Note: difference betwwen ax and plt method are the get_ and set_ suffix
        ax = plt.gca()
    if logscale:
        ax.set_xscale('log'); ax.set_yscale('log')
    if title:
        ax.set_title(title)

    fit = np.polyfit(np.log(d), np.log(dc), deg=1)
    #label = 'power %.2f' % abs(fit[0])
    label = ''
    ax.plot(d, np.exp(fit[0] *np.log(d) + fit[1]), 'g--', label=label % fit[1])
    if label:
        leg = ax.legend(loc='upper right',prop={'size':10})

    if scatter:
        ax.scatter(d,dc,c='b',marker='o')
        #plt.scatter(ba_x,ba_y,c='r',marker='s',s=50)

    if spec is True:
        ax.set_xlim(left=1)
        ax.set_ylim((.9,1e3))
        ax.set_xlabel('Degree')
        ax.set_ylabel('Counts')

def plot_degree_poly_l(Y):
    """ Same than plot_degree_poly, but for a list of random graph ie plot errorbar."""
    x, y, yerr = random_degree(Y)

    plt.xscale('log'); plt.yscale('log')

    fit = np.polyfit(np.log(x), np.log(y), deg=1)
    plt.plot(x,np.exp(fit[0] *np.log(x) + fit[1]), 'm:', label='model power %.2f' % fit[1])
    leg = plt.legend(loc='upper right',prop={'size':10})

    plt.errorbar(x, y, yerr=yerr, fmt='o')

    plt.xlim(left=1)
    plt.ylim((.9,1e3))
    plt.xlabel('Degree'); plt.ylabel('Counts')

def plot_degree_2(P, logscale=False, colors=False, line=False, ax=None, title=None):
    """ Plot degree distribution for different configuration"""
    if ax is None:
        # Note: difference betwwen ax and plt method are the get_ and set_ suffix
        ax = plt.gca()

    x, y, yerr = P
    y = np.ma.array(y)
    for i, v in enumerate(y):
        if v == 0:
            y[i] = np.ma.masked
        else:
            break

    #x = x[::2]
    #y = y[::2]
    #yerr = yerr[::2]

    c = next(_colors) if colors else 'b'
    m = next(_markers) if colors else 'o'
    l = '--' if line else None

    if yerr is None:
        ax.scatter(x, y, c=c, marker=m)
        if line:
            ax.plot(x, y, c=c, marker=m, ls=l)
    else:
        ax.errorbar(x, y, yerr, c=c, fmt=m, ls=l)

    min_d, max_d = min(x), max(x)

    if logscale:
        ax.set_xscale('log'); ax.set_yscale('log')
        # Ensure that the ticks will be visbile (ie larger than in los step)
        #logspace = 10**np.arange(6)
        #lim =  np.searchsorted(logspace,min_d )
        #if lim == np.searchsorted(logspace,max_d ):
        #    min_d = logspace[lim-1]
        #    max_d = logspace[lim]
    if title:
        ax.set_title(title)

    #ax.set_xlim((min_d, max_d+10))
    #ax.set_xlim(left=1)
    ax.set_ylim((.9,1e3))
    ax.set_xlabel('Degree'); ax.set_ylabel('Counts')

##########################
### Graph/Matrix Drawing
##########################

def draw_boundary(mat, clusters):
    """ Clusters: list of membership or list of boudary """
    # sorted operation in reorder and clusters could optimized (redondancy)
    N = mat.shape[0]
    if len(clusters) < N:
        B = clusters
        K = len(B) - 1
        B = B[1:-1]
    else:
        mat = reorder_mat(mat, clusters)
        B, __ = clusters_hist(clusters)
        B = np.cumsum(B)[:-1]

    K = len(B)
    c = np.arange(mat.max()+1, mat.max()+1+K+1)
    w = int(0.005 * N)
    print ('drawing boundary %s' % B)
    for i, b in enumerate(B):
        mat[b-w:b+w, :] = c[i]
        mat[:, b-w:b+w] = c[i]
    return mat

def draw_graph_spring(y, clusters='blue', ns=30):
    G = nxG(y)

    plt.figure()
    nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_color = clusters, node_size=30, with_labels=False)

def draw_blocks(comm):
    #nx.draw(H,G.position,
    #        node_size=[G.population[v] for v in H],
    #        node_color=node_color,
    #        with_labels=False)
    blocks = comm.get('block_hist')
    ties = comm.get('block_ties')

    blocks = 2*blocks / np.linalg.norm(blocks)
    max_n = max(blocks)

    G = nx.Graph(nodesep=0.7)
    u_colors.reset()
    ind_color = np.arange(0, len(blocks)**2, 2) % len(u_colors.seq)
    #ind_color = np.diag(np.arange(len(blocks)**2).reshape([len(blocks)]*2)) % len(u_colors.seq)
    colors = np.array(u_colors.seq)[ind_color]

    # if sorted
    sorted_blocks, sorted_ind = zip(*sorted( zip(blocks, range(len(blocks))) , reverse=True))
    for l, s in enumerate(sorted_blocks):
        if s == 0:
            continue
        G.add_node(int(l), width=s, height=s, fillcolor=colors[l], style='filled')

    max_t = max(ties, key=lambda x:x[1])[1]
    if max_t > max_n:
        scale = np.exp(2) * float(max_n) / max_t

    for l, s in ties:
        i, j = l
        # if sorted
        i = sorted_ind.index(int(i))
        j = sorted_ind.index(int(j))
        G.add_edge(i, j, penwidth = s * scale)

    return write_dot(G, 'graph.dot')

try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydotplus
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        lgg.debug("Networks needs Graphviz and either PyGraphviz or PyDotPlus")
def draw_graph_circular(y, clusters='blue', ns=30):
    G = nxG(y)
    pos = graphviz_layout(G, prog='twopi', args='')
    plt.figure()
    nx.draw(G, pos, node_size=ns, alpha=0.8, node_color=clusters, with_labels=False)
    plt.axis('equal')

def draw_graph_spectral(y, clusters='blue', ns=30):
    G = nxG(y)
    pos = graphviz_layout(G, prog='twopi', args='')
    plt.figure()
    nx.draw_spectral(G, cmap = plt.get_cmap('jet'), node_color = clusters, node_size=30, with_labels=False)

def adjblocks(Y, clusters=None, title=''):
    """ Make a colormap image of a matrix
        :key Y: the matrix to be used for the colormap.
    """
    # Artefact
    #np.fill_diagonal(Y, 0)

    plt.figure()
    if clusters is None:
        plt.axis('off')
        cmap = 'Greys'
        norm = None
        y = Y
    else:
        plt.axis('on')
        y = reorder_mat(Y, clusters, reverse=True)
        hist, label = clusters_hist(clusters)
        # Colors Setup
        u_colors.reset()
        #cmap = Colors.ListedColormap(['white','black']+ u_colors.seq[:len(hist)**2])
        cmap = Colors.ListedColormap(['#000000', '#FFFFFF']+ u_colors.seq[:len(hist)**2])
        bounds = np.arange(len(hist)**2+2)
        norm = Colors.BoundaryNorm(bounds, cmap.N)
        # Iterate over blockmodel
        for k, count_k in enumerate(hist):
            for l, count_l in enumerate(hist):
                # Draw a colored square (not white and black)
                topleft =  (hist[:k].sum(), hist[:l].sum())
                w = int(0.01 * y.shape[0])
                # write on place
                draw_square(y, k+l+2, topleft, count_k, count_l, w)

    implt = plt.imshow(y, cmap=cmap, norm=norm,
                       clim=(np.amin(y), np.amax(y)),
                       interpolation='nearest',)
                       #origin='upper') # change shape !
    #plt.colorbar()
    plt.title(title)
    #plt.savefig(filename, fig=fig, facecolor='white', edgecolor='black')

def adjshow(Y, title='', fig=True, ax=None, colorbar=False):
    if fig is True and ax is None:
        plt.figure()
    if ax is None:
        # Note: difference betwwen ax and plt method are the get_ and set_ suffix
        ax = plt.gca()

    ax.axis('off')
    #cmap = 'Greys'
    cmap = plt.cm.hot
    if Y.shape[0] != Y.shape[1]:
        aspect = 'auto'
    else:
        aspect = None

    img = ax.imshow(Y, cmap=cmap,
                    aspect=aspect,
                    interpolation='None')
    ax.set_title(title)
    if colorbar:
        plt.colorbar(img, ax=ax)

def adjshow_4(Y,title=[], pixelspervalue=20):
    minvalue = np.amin(Y)
    maxvalue = np.amax(Y)
    cmap = plt.cm.hot

    fig = plt.figure()
    plt.subplot(2,2,1)
    plt.axis('off')
    implot = plt.imshow(Y[0], cmap=cmap, clim=(minvalue, maxvalue), interpolation='nearest')
    plt.title(title[0])

    plt.subplot(2,2,2)
    plt.axis('off')
    implot = plt.imshow(Y[1], cmap=cmap, clim=(minvalue, maxvalue), interpolation='nearest')
    plt.title(title[1])

    plt.subplot(2,2,3)
    plt.axis('off')
    implot = plt.imshow(Y[2], cmap=cmap, clim=(minvalue, maxvalue), interpolation='nearest')
    plt.title(title[2])

    plt.subplot(2,2,4)
    plt.axis('off')
    implot = plt.imshow(Y[3], cmap=cmap, clim=(minvalue, maxvalue), interpolation='nearest')
    plt.title(title[3])

    plt.draw()


##########################
### Curve Plot
##########################

# @deprecated
def plot_csv(target_dirs='', columns=0, sep=' ', separate=False, title=None, twin=False, iter_max=None):
    if type(columns) is not list:
        columns = [columns]

    if type(target_dirs) is not list:
        target_dirs = [target_dirs]

    title = title or 'Inference'
    xlabel = 'Iterations'

    prop_set = get_expe_file_set_prop(target_dirs)
    id_plot = 0
    old_prop = 0
    fig = None
    _Ks = None
    is_nonparam = False
    #if separate is False:
    #    fig = plt.figure()
    #    ax1 = fig.add_subplot(111)
    #    ax1.set_xlabel(xlabel)
    #    ax1.set_title(title)

    for i, target_dir in enumerate(target_dirs):

        filen = os.path.join(os.path.dirname(__file__), "../data/", target_dir)
        print ('plotting in %s ' % (filen, ))
        with open(filen) as f:
            data = f.read()

        data = filter(None, data.split('\n'))
        if iter_max:
            data = data[:iter_max]
        data = [re.sub("\s\s+" , " ", x.strip()) for l,x in enumerate(data) if not x.startswith(('#', '%'))]
        #data = [x.strip() for x in data if not x.startswith(('#', '%'))]

        prop = get_expe_file_prop(target_dir)
        for column in columns:
            if type(column) is str:
                column = csv_row(column)

            ### Optionnal row...?%
            if separate is False:
                if fig is None:
                    fig = plt.figure()
                plt.subplot(1,1,1)
                plt.title(title)
                plt.xlabel(xlabel)
                ax1 = plt.gca()
            elif separate is True:
                fig = plt.figure()
                plt.title(title)
                plt.xlabel(xlabel)
                ax1 = plt.gca()
            elif type(separate) is int:
                if i % (separate*2) == 0:
                    fig = plt.figure()
                    plt.subplot(1,2,1)
                if i % (separate*2) == 2:
                    plt.subplot(1,2,2)
                plt.title(title)
                plt.xlabel(xlabel)
                ax1 = plt.gca()
            elif type(separate) is str:
                if fig is None:
                    fig = plt.figure()
                if prop['N'] != old_prop:
                    id_plot += 1
                old_prop = prop['N']
                plt.subplot(1,prop_set['N'],id_plot)
                plt.title(title)
                plt.xlabel(xlabel)
                ax1 = plt.gca()

            ll_y = [row.split(sep)[column] for row in data]
            ll_y = np.ma.masked_invalid(np.array(ll_y, dtype='float'))

            Ks = [int(float(row.split(sep)[csv_row('K')])) for row in data]
            Ks = np.ma.masked_invalid(np.array(Ks, dtype='int'))

            ylabel, label = tag_from_csv(column)
            ax1.set_ylabel(ylabel)

            model_label = target_dir.split('/')[-1]
            #label = target_dir + ' ' + label
            label = target_dir.split('/')[-3] +' '+ model_label
            if model_label.startswith(('ilda', 'immsb')):
                is_nonparam = True
                label += ' K -> %d' % (float(Ks[-1]))

            ax1.plot(ll_y, marker=next(_markers), label=label)
            leg = ax1.legend(loc=1,prop={'size':10})

            if not twin:
                continue
            if is_nonparam or _Ks is None:
                _Ks = Ks
            Ks = _Ks
            ax2 = ax1.twinx()
            ax2.plot(Ks, marker='*')

        #plt.savefig('../results/debug1/%s.pdf' % (prop['corpus']))
    plt.draw()

if __name__ ==  '__main__':
    block = True

    display(block)

