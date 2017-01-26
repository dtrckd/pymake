# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from utils import *

import logging
lgg = logging.getLogger('root')

##########################
### Stochastic Process
##########################

def lognormalize(x):
    a = np.logaddexp.reduce(x)
    return np.exp(x - a)

def categorical(params):
    return np.where(np.random.multinomial(1, params) == 1)[0]

def bernoulli(param, size=1):
    return np.random.binomial(1, param, size=size)

### Power law distribution generator
def random_powerlaw(alpha, x_min, size=1):
    ### Discrete
    alpha = float(alpha)
    u = np.random.random(size)
    x = (x_min-0.5)*(1-u)**(-1/(alpha-1))+0.5
    return np.floor(x)

### A stick breakink process, truncated at K components.
def gem(gmma, K):
    sb = np.empty(K)
    cut = np.random.beta(1, gmma, size=K)
    for k in range(K):
        sb[k] = cut[k] * cut[0:k].prod()
    return sb

##########################
### Means and Norms
##########################

### Weighted  means
def wmean(a, w, mean='geometric'):
    if mean == 'geometric':
        kernel = lambda x : np.log(x)
        out = lambda x : np.exp(x)
    elif mean == 'arithmetic':
        kernel = lambda x : x
        out = lambda x : x
    elif mean == 'harmonic':
        num = np.sum(w)
        denom = np.sum(np.asarray(w) / np.asarray(a))
        return num / denom
    else:
        raise NotImplementedError('Mean Unknwow: %s' % mean)

    num = np.sum(np.asarray(w) * kernel(np.asarray(a)))
    denom = np.sum(np.asarray(w))
    return out(num / denom)

##########################
### Matrix/Image Operation
##########################
from scipy import ndimage

def draw_square(mat, value, topleft, l, L, w=0):
    tl = topleft

    # Vertical draw
    mat[tl[0]:tl[0]+l, tl[1]:tl[1]+w] = value
    mat[tl[0]:tl[0]+l, tl[1]+L-w:tl[1]+L] = value
    # Horizontal draw
    mat[tl[0]:tl[0]+w, tl[1]:tl[1]+L] = value
    mat[tl[0]+l-w:tl[0]+l, tl[1]:tl[1]+L] = value
    return mat

def dilate(y):
    mask = ndimage.generate_binary_structure(2, 2)
    y_f = ndimage.binary_dilation(y, structure=mask).astype(y.dtype)
    return y_f


##########################
### Array routine  Operation
##########################
from collections import Counter

def sorted_perm(a, label=None, reverse=False):
    """ return sorted $a and the induced permutation.
        Inplace operation """
    # np.asarray applied this tuple lead to error, if label is string
    # because a should be used as elementwise comparison
    if label is None:
        label = np.arange(a.shape[0])
    hist, label = zip(*sorted(zip(a, label), reverse=reverse))
    hist = np.asarray(hist)
    label = np.asarray(label)
    return hist, label

def degree_hist_to_list(d, dc):
    degree = np.repeat(np.round(d).astype(int), np.round(dc).astype(int))
    return degree


def clusters_hist(clusters, labels=None, remove_empty=True):
    """ return non empty clusters histogramm sorted.

        parameters
        ---------
        clusters: np.array
            array of clusters membership of data.

        returns
        -------
        hist: np.array
            count of element by clusters (decrasing hist)
        label: np.array
            label of the cluster aligned with hist
    """
    block_hist = np.bincount(clusters)
    if labels is None:
        labels = range(len(block_hist))

    hist, label = sorted_perm(block_hist, labels, reverse=True)

    if remove_empty is True:
        null_classes = (hist == 0).sum()
        if null_classes > 0:
            hist = hist[:-null_classes]; label = label[:-null_classes]

    return hist, label

from .utils import nxG
import networkx as nx
def adj_to_degree(y):
    # @debug: dont' call nxG or do a native integration !

    # To convert normalized degrees to raw degrees
    #ba_c = {k:int(v*(len(ba_g)-1)) for k,v in ba_c.iteritems()}
    G = nxG(y)
    #degree = sorted(nx.degree(G).values(), reverse=True)

    #ba_c = nx.degree_centrality(G)
    return nx.degree(G)

def degree_hist(_degree):
    degree = _degree.values() if type(_degree) is dict else _degree
    bac = dict(Counter(degree))

    #ba_x,ba_y = log_binning(bac,50)
    d = np.array(list(bac.keys()))  # Degrees
    dc = np.array(list(bac.values())) # Degree counts

    if d[0] == 0:
        lgg.debug('%d unconnected vertex' % dc[0])
        d = d[1:]
        dc = dc[1:]


    if len(d) != 0:
        d, dc = zip(*sorted(zip(d, dc)))
    return np.round(d), np.round(dc)

def random_degree(Y, params=None):
    _X = []
    _Y = []
    N = Y[0].shape[0]
    size = []
    for y in Y:
        ba_c = adj_to_degree(y)
        d, dc = degree_hist(ba_c)

        _X.append(d)
        _Y.append(dc)
        size.append(len(_Y[-1]))

    min_d = min(size)
    for i, v in enumerate(_Y):
        if len(v) > min_d:
            _X[i] = _X[i][:min_d]
            _Y[i] = _Y[i][:min_d]

    X = np.array(_X)
    Y = np.array(_Y)
    x = X.mean(0)
    y = Y.mean(0)
    yerr = Y.std(0)
    return np.round(x), np.round(y), yerr


def reorder_mat(y, clusters, reverse=True, labels=False):
    """Reorder the  matrix according the clusters membership
        @Debug: square matrix
    """
    assert(y.shape[0] == y.shape[1] == len(clusters))
    if reverse is True:
        hist, label = clusters_hist(clusters)
        sorted_clusters = np.empty_like(clusters)
        for i, k in enumerate(label):
            if i != k:
                sorted_clusters[clusters == k] = i
    else:
        sorted_clusters = clusters

    N = y.shape[0]
    nodelist = [k[0] for k in sorted(zip(range(N), sorted_clusters),
                                     key=lambda k: k[1])]

    y_r = y[nodelist, :][:, nodelist]
    if labels is True:
        return y_r, nodelist
    else:
        return y_r

def shiftpos(arr, fr, to, axis=0):
    """ Move element In-Place, shifting backward (or forward) others """
    if fr == to: return
    x = arr.T if axis == 1 else arr
    tmp = x[fr].copy()
    if fr > to:
        x[to+1:fr+1] = x[to:fr]
    else:
        x[fr:to] = x[fr+1:to+1]
    x[to] = tmp


##########################
### Colors Operation
##########################
import math

def floatRgb(mag, cmin, cmax):
	""" Return a tuple of floats between 0 and 1 for the red, green and
		blue amplitudes.
	"""

	try:
		# normalize to [0,1]
		x = float(mag-cmin)/float(cmax-cmin)
	except:
		# cmax = cmin
		x = 0.5
	blue = min((max((4*(0.75-x), 0.)), 1.))
	red  = min((max((4*(x-0.25), 0.)), 1.))
	green= min((max((4*math.fabs(x-0.5)-1., 0.)), 1.))
	return (red, green, blue)

def strRgb(mag, cmin, cmax):
   """ Return a tuple of strings to be used in Tk plots.
   """

   red, green, blue = floatRgb(mag, cmin, cmax)
   return "#%02x%02x%02x" % (red*255, green*255, blue*255)

def rgb(mag, cmin, cmax):
   """ Return a tuple of integers to be used in AWT/Java plots.
   """

   red, green, blue = floatRgb(mag, cmin, cmax)
   return (int(red*255), int(green*255), int(blue*255))

def htmlRgb(mag, cmin, cmax):
   """ Return a tuple of strings to be used in HTML documents.
   """
   return "#%02x%02x%02x"%rgb(mag, cmin, cmax)

