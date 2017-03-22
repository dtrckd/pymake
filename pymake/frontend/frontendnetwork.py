# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os
import itertools
from string import Template

from numpy import ma
import numpy as np
import networkx as nx
try:
    import community as pylouvain
except:
    pass

from .frontend import DataBase
from pymake.util.utils import parse_file_conf
from pymake.util.math import *

def getClique(N=100, K=4):
    from scipy.linalg import block_diag
    b = []
    for k in range(K):
        n = N // K
        b.append(np.ones((n,n), int))

    C = block_diag(*b)
    return C

### @Issue42: fronteNetwork should be imported fron frontend
### =====> : resolve this with @class_method (from_hardrive etc...)

class frontendNetwork(DataBase):
    """ Frontend for network data.
        Symmetric network support.
    """

    RANDOM_CORPUS = ('clique', 'alternate', 'BA')
    _selfloop = True

    def __init__(self, expe=dict(), load=False):
        self.bdir = 'networks'
        super(frontendNetwork, self).__init__(expe, load)

    def load_data(self, corpus_name=None, randomize=False):
        """ Load data according to different scheme,
            by order of priority (if several specification in settings)
            * Corpus from random generator
            * Corpus from file dataset
        """
        if corpus_name is not None:
            self.update_spec(**{'corpus_name': corpus_name})
        else:
            corpus_name = self.corpus_name
        self.make_output_path()

        if self.corpus_name.startswith(self.RANDOM_CORPUS):
            data = self.random_corpus(corpus_name)
        else:
            self.make_output_path()
            data = self._get_corpus(corpus_name)

        self.update_data(data)

        # For Gof smothness
        # error in degree_ check ?
        if self.has_selfloop():
            np.fill_diagonal(self.data, 1)

        if randomize:
            self.shuffle_node()
        return self.data

    def _get_corpus(self, corpus_name):
        """ @debug Be smarter, has some database strategy.
            Redirect to correct path depending on the corpus_name
        """

        # DB integration ?
        if corpus_name.startswith(('generator', 'graph')):
            format = 'graph'
        elif corpus_name in ('bench1'):
            raise NotImplementedError()
        elif corpus_name.startswith('facebook'):
            format = 'edges'
        elif corpus_name in ('manufacturing',):
            format = 'csv'
        elif corpus_name in ('fb_uc', 'emaileu'):
            format = 'txt'
        elif corpus_name in ('blogs','propro', 'euroroad'):
            format = 'dat'
        else:
            raise ValueError('Which corpus to Load; %s ?' % corpus_name)

        data = self.networkloader(corpus_name, format)

        for a in ('features', 'clusters'):
            if not hasattr(self, a):
                setattr(self, a, None)

        return data

    def shuffle_node(self):
        """ Shuffle rows and columns of data """
        N, M = self.data.shape
        nodes_list = [np.random.permutation(N), np.random.permutation(M)]
        self.reorder_node(nodes_list)

    def reorder_node(self, nodes_l):
        """ Subsample the data with reordoring of rows and columns """
        self.data = self.data[nodes_l[0], :][:, nodes_l[1]]
        # Track the original nodes
        self.nodes_list = [self.nodes_list[0][nodes_l[0]], self.nodes_list[1][nodes_l[1]]]

    def sample(self, N=None, symmetric=False, randomize=False):
        """ Write self ! """
        N = N or self.getN()

        if N == 'all':
            N = self.data.shape[0]
        else:
            N = int(N)

        # Can't get why modification inside self.nodes_list is not propagated ?
        if randomize is True:
            nodes_list = [np.random.permutation(N), np.random.permutation(N)]
            self.reorder_node(nodes_list)

        if N < self.data.shape[0]:
            self.data = self.data[:N, :N]
            self.update_data(self.data)
        return self.data

    def update_data(self, data):
        ''' node list order will be lost '''
        self.data = data
        N, M = self.data.shape
        self.N = N
        self.nodes_list = [np.arange(N), np.arange(M)]

        if hasattr(self, 'features') and self.features is not None:
            self.features = self.features[:N]

    def get_masked(self, percent_hole, diag_off=1):
        """ Construct a random mask.
            Random training set on 20% on Data / debug5 - debug11 -- Unbalanced
        """
        percent_hole = float(percent_hole)
        if percent_hole >= 1:
            percent_hole = percent_hole / 100.0
        elif 0 <= percent_hole < 1:
            pass
        else:
            raise ValueError('cross validation ration not understood : %s' % percent_hole)

        data = self.data
        if type(data) is np.ndarray:
            #self.data_mat = sp.sparse.csr_matrix(data)
            pass
        else:
            raise NotImplementedError('type %s unknow as corpus' % type(data))

        n = int(data.size * percent_hole)
        mask_index = np.unravel_index(np.random.permutation(data.size)[:n], data.shape)
        mask = np.zeros(data.shape, dtype=int)
        mask[mask_index] = 1

        if self.is_symmetric():
            mask = np.tril(mask) + np.tril(mask, -1).T

        data_ma = ma.array(data, mask=mask)
        if diag_off == 1:
            np.fill_diagonal(data_ma, ma.masked)

        return data_ma

    def set_masked(self, percent_hole, diag_off=1):
        self.data_ma = self.get_masked(percent_hole, diag_off)
        return self.data_ma

    def get_masked_1(self, percent_hole, diag_off=1):
        ''' Construct Mask nased on the proportion of 1/links.
            Random training set on 20% on Data vertex (0.2 * data == 1) / debug6 - debug 10 -- Balanced
            '''
        data = self.data
        if type(data) is np.ndarray:
            #self.data_mat = sp.sparse.csr_matrix(data)
            pass
        else:
            raise NotImplementedError('type %s unknow as corpus' % type(data))

        # Correponding Index
        _0 = np.array(zip(*np.where(data == 0)))
        _1 = np.array(zip(*np.where(data == 1)))
        n = int(len(_1) * percent_hole)
        # Choice of Index
        n_0 = _0[np.random.choice(len(_0), n, replace=False)]
        n_1 = _1[np.random.choice(len(_1), n, replace=False)]
        # Corresponding Mask
        mask_index = zip(*(np.concatenate((n_0, n_1))))
        mask = np.zeros(data.shape, dtype=int)
        mask[mask_index] = 1

        if self.is_symmetric():
            mask = np.tril(mask) + np.tril(mask, -1).T

        data_ma = ma.array(data, mask=mask)
        if diag_off == 1:
            np.fill_diagonal(data_ma, ma.masked)

        return data_ma

    def is_symmetric(self, update=False):
        if update or not hasattr(self, 'symmetric'):
            self.symmetric = (self.data == self.data.T).all()
        return self.symmetric

    def random_corpus(self, rnd):
        N = self.getN()

        if rnd == 'uniform':
            data = np.random.randint(0, 2, (N, N))
            #np.fill_diagonal(data, 1)
        elif rnd.startswith('clique'):
            try :
                K = int(rnd[len('clique'):])
            except ValueError:
                K = 42
            data = getClique(N, K=K)
            #Data = nx.adjacency_matrix(G, np.random.permutation(range(N))).A
        elif rnd in ('BA', 'barabasi-albert'):
            data = nx.adjacency_matrix(nx.barabasi_albert_graph(N, m=13) ).A
        elif rnd ==  'alternate':
            #data = np.empty((N,N),int)
            data = np.zeros((N,N), int)
            type_rd = 2
            if type_rd == 1:
                # degree alternating with frequency fr
                fr = 3
                data[:, ::fr] = 1
            elif type_rd == 2:
                # degree equal
                data[:, ::2] = 1
                data[::2] = np.roll(data[::2], 1)
            return data
        else:
            raise NotImplementedError()

        return data

    def networkloader(self, corpus_name, format):
        """ Load pickle or parse data.
            Format is understanding for parsing.
            """
        data = None
        fn = os.path.join(self.basedir, corpus_name)
        if self._load_data and os.path.isfile(fn+'.pk'):
            try:
                data = self.load(fn)
            except Exception as e:
                lgg.error('Error : %s on %s' % (e, fn+'.pk'))
                data = None
        if data is None:
            ext = format
            fn = os.path.join(self.basedir, corpus_name +'.'+ ext)
            if ext == 'graph':
                fn = os.path.join(self.basedir, 't0.graph')
                data = self.parse_dancer(fn)
            elif ext == 'edges':
                fn = os.path.join(self.basedir, '0.edges')
                data = self.parse_edges(fn)
                # NotImplemented
            elif ext in ('txt'):
                data = self.parse_tnet(fn)
            elif ext == 'csv':
                #elif os.path.basename(fn) == 'manufacturing.csv':
                data = self.parse_csv(fn)
            elif ext == 'dat':
                data = self.parse_dat(fn)
            else:
                raise ValueError('extension of network data unknown')

        if self._save_data:
            self.save(data, fn[:-len('.'+ext)])

        return data

    def parse_tnet(self, fn):
        sep = ' '
        lgg.debug('opening file: %s' % fn)
        with open(fn) as f:
            content = f.read()
        lines = filter(None, content.split('\n'))
        edges = [l.strip().split(sep)[-3:-1] for l in lines]
        edges = np.array([ (e[0], e[1]) for e in edges], dtype=int) -1
        N = edges.max() +1
        #N = max(list(itertools.chain(*edges))) + 1

        g = np.zeros((N,N), dtype=int)
        g[tuple(edges.T)] = 1
        return g

    def parse_csv(self, fn):
        sep = ';'
        lgg.debug('opening file: %s' % fn)
        with open(fn, 'r') as f:
            content = f.read()
        lines = list(filter(None, content.split('\n')))[1:]
        edges = [l.strip().split(sep)[0:2] for l in lines]
        edges = np.array([ (e[0], e[1]) for e in edges], dtype=int) -1
        N = edges.max() +1
        #N = max(list(itertools.chain(*edges))) + 1

        g = np.zeros((N,N), dtype=int)
        g[tuple(edges.T)] = 1
        return g

    def parse_dancer(self, fn, sep=';'):
        """ Parse Network data depending on type/extension """
        lgg.debug('opening file: %s' % fn)
        f = open(fn, 'r')
        data = []
        inside = {'vertices':False, 'edges':False }
        clusters = []
        features = []
        for line in f:
            if line.startswith('# Vertices') or inside['vertices']:
                if not inside['vertices']:
                    inside['vertices'] = True
                    N = 0
                    continue
                if line.startswith('#') or not line.strip() :
                    inside['vertices'] = False # break
                else:
                    # Parsing assignation
                    elements = line.strip().split(sep)
                    clust = int(elements[-1])
                    feats = list(map(float, elements[-2].split('|')))
                    clusters.append(clust)
                    features.append(feats)
                    N += 1
            elif line.startswith('# Edges') or inside['edges']:
                if not inside['edges']:
                    inside['edges'] = True
                    continue
                if line.startswith('#') or not line.strip() :
                    inside['edges'] = False # break
                else:
                    # Parsing assignation
                    data.append( line.strip() )
        f.close()

        edges = np.array([tuple(row.split(sep)) for row in data]).astype(int)
        g = np.zeros((N,N), dtype=int)
        g[[e[0] for e in edges], [e[1] for e in edges]] = 1
        g[[e[1] for e in edges], [e[0] for e in edges]] = 1
        # ?! .T

        try:
            parameters = parse_file_conf(os.path.join(os.path.dirname(fn), 'parameters'))
            parameters['devs'] = list(map(float, parameters['devs'].split(sep)))
        except IOError:
            parameters = {}
        finally:
            self.parameters_ = parameters

        self.clusters = clusters
        self.features = np.array(features)
        return g

    def parse_dat(self, fn, sep=' '):
        """ Parse Network data depending on type/extension """
        lgg.debug('opening file: %s' % fn)
        f = open(fn, 'rb')
        data = []
        inside = {'vertices':False, 'edges':False }
        clusters = []
        features = []
        for _line in f:
            line = _line.strip().decode('utf8') #python 2..
            if line.startswith(('ROW LABELS:', '*vertices')) or inside['vertices']:
                if not inside['vertices']:
                    inside['vertices'] = True
                    continue
                if line.startswith('#') or not line.strip():
                    inside['vertices'] = False # break
                elif line.startswith(('DATA','*edges' )):
                    inside['vertices'] = False # break
                    inside['edges'] = True
                else:
                    # todo if needed
                    continue
            elif line.startswith(('DATA','*edges' )) or inside['edges']:
                if not inside['edges']:
                    inside['edges'] = True # break
                    continue
                if line.startswith('#') or not line.strip() or len(line.split(sep)) < 2 :
                    inside['edges'] = False
                else:
                    # Parsing assignation
                    data.append( line.strip() )
        f.close()

        row_size = len(data[0].split(sep))
        edges = np.array([tuple(row.split(sep)) for row in data]).astype(int)-1
        if row_size == 2:
            # like .txt
            pass
        elif row_size == 3:
            # has zeros...
            # take edges if non zeros edges, and remove last column (edges]
            edges = edges[ edges.T[-1] > -1 ][:, :-1] # -1, edges start from 1
            edges = np.array([(e[0],e[1]) for e in edges]).astype(int)
        else:
            raise NotImplementedError

        N = edges.max() +1
        g = np.zeros((N,N), dtype=int)
        g[tuple(edges.T)] = 1
        return g

    def _old_communities_analysis(self):
        clusters = self.clusters
        if clusters is None:
            return None
        data = self.data
        symmetric = self.is_symmetric()
        community_distribution = list(np.bincount(clusters))

        local_attach = {}
        for n, c in enumerate(clusters):
            comm = str(c)
            local = local_attach.get(comm, [])
            degree_n = data[n,:][clusters == c].sum()
            if not symmetric:
                degree_n += data[:, n][clusters == c].sum()
            local.append(degree_n)
            local_attach[comm] = local

        return community_distribution, local_attach, clusters

    # used by (obsolete) zipf.py
    def communities_analysis(self, *args, **kwargs):
        from pymake.util.algo import adj_to_degree # Circular import bug inthetop
        clusters = self.clusters
        if clusters is None:
            return None
        data = self.data
        symmetric = self.is_symmetric()
        community_distribution = list(np.bincount(clusters))
        block_hist = np.bincount(clusters)

        local_degree = {}
        if symmetric:
            k_perm = np.unique(list( map(list, map(set, itertools.product(np.unique(clusters) , repeat=2)))))
        else:
            k_perm = itertools.product(np.unique(clusters) , repeat=2)

        for c in k_perm:
            if type(c) in (np.float64, np.int64):
                # one clusters (as it appears for real with max assignment
                l = k = c
            elif  len(c) == 2:
                # Stochastic Equivalence (extra class bind
                k, l = c
            else:
                # Comunnities (intra class bind)
                k = l = c.pop()
            comm = (str(k), str(l))
            local = local_degree.get(comm, [])

            C = np.tile(clusters, (data.shape[0],1))
            y_c = data * ((C==k) & (C.T==l))
            if y_c.size > 0:
                local_degree[comm] = adj_to_degree(y_c).values()

            # Summing False !
            #for n in np.arange(data.shape[0]))[clusters == k]:
            #    degree_n = data[n,:][(clusters == k) == (clusters == l)].sum()
            #    if not symmetric:
            #        degree_n = data[n,:][(clusters == k) == (clusters == l)].sum()
            #    local.append(degree_n)
            #local_degree[comm] = local

        return {'local_degree':local_degree,
                'clusters': np.asarray(clusters),
                'block_hist': block_hist,
                'size': len(block_hist)}


    def getG(self):
        if not hasattr(self, 'G'):
            if self.is_symmetric():
                # Undirected Graph
                typeG = nx.Graph()
            else:
                # Directed Graph
                typeG = nx.DiGraph()
            self.G = nx.from_numpy_matrix(self.data, create_using=typeG)
            #self.G = nx.from_scipy_sparse_matrix(self.data, typeG)
        return self.G

    def to_directed(self):
        ''' Return self verion of graph wehre all links are flatened '''
        if self.is_symmetric():
            return self.getG()
        else:
            # nx to_undirected nedd a linkks in both side.
            return nx.from_numpy_matrix(self.data, create_using=nx.Graph())

    #
    # Get Statistics
    #

    def nodes(self):
        g = self.getG()
        return g.number_of_nodes()

    def edges(self):
        g = self.getG()
        return g.number_of_edges()

    def diameter(self):
        g = self.getG()
        try:
            diameter = nx.diameter(g)
        except:
            diameter = None
        return diameter

    def density(self):
        g = self.getG()
        return nx.density(g)

    def modularity(self):
        part =  self.get_partition()
        if not part:
            return None
        g = self.getG()
        try:
            modul = pylouvain.modularity(part, g)
        except NameError:
            lgg.error('python-louvain) library is not installed \n \
                      Modularity can\'t be computed ')
            modul = None
        return modul

    def clustering_coefficient(self):
        g = self.getG()
        try:
            cc = nx.average_clustering(g)
        except:
            cc = None
        return cc

    def getN(self):
        if hasattr(self, 'N'):
            return self.N

        N = str(self.expe['N'])
        if N.isdigit():
            N = int(N)
        elif N.lower() in ('all', 'false', 'none'):
            N = 'all'
        else:
            raise TypeError('Size of data no set (-n)')

        self.N = N
        return self.N

    #def louvain_feature(self):
    #    get the louvain modularity
    #    and the feature for local analysis

    def degree(self):
        g = self.getG()
        return nx.degree(g)

    def degree_histogram(self):
        g = self.getG()
        return nx.degree_histogram(g)

    def get_nfeat(self):
        nfeat = self.data.max() + 1
        if nfeat == 1:
            lgg.warn('Warning, only zeros in adjacency matrix...')
            nfeat = 2
        return nfeat

    def get_nnz(self):
        ''' len of tokens '''
        size =  sp.special.binom(self.getN(), 2)
        if not self.is_symmetric():
            size *= 2

        if self.has_selfloop():
            size += sekf.getN()

        return size

    def ma_nnz(self):
        return len(self.data_ma.compressed())

    # Contains the index of nodes with who it interact.
    # @debug no more true for bipartite networks
    def ma_dims(self):
        #data_dims = np.vectorize(len)(self.data)
        #data_dims = [r.count() for r in self.data_ma]
        data_dims = []
        for i in range(self.data_ma.shape[0]):
            data_dims.append(self.data_ma[i,:].count() + self.data_ma[:,i].count())
        return np.array(data_dims, dtype=int)

    def has_selfloop(self):
        return self._selfloop

    def get_params(self):
        clusters = self.get_clusters()
        K = max(clusters)+1
        N = len(clusters)
        theta = np.zeros((N,K))
        theta[np.arange(N),clusters] = 1
        return theta, None

    def get_clusters(self):
        return self.clusters

    def get_partition(self, clusters=None):
        clusters = self.clusters or clusters
        if not clusters:
            return {}
        N = len(clusters)
        return dict(zip(*[np.arange(N), clusters]))

    def clusters_len(self):
        clusters = self.get_clusters()
        if not clusters:
            return None
        else:
            return max(clusters)+1

    def get_data_prop(self):
        prop =  super(frontendNetwork, self).get_data_prop()

        if self.is_symmetric():
            nnz = np.triu(self.data).sum()
        else:
            nnz = self.data.sum()

        _nnz = self.data.sum(axis=1)
        d = {'instances': self.data.shape[1],
               'nnz': nnz,
               'nnz_mean': _nnz.mean(),
               'nnz_var': _nnz.var(),
               'density': self.density(),
               'diameter': self.diameter(),
               'clustering_coef': self.clustering_coefficient(),
               'modularity': self.modularity(),
               'communities': self.clusters_len(),
               'features': self.get_nfeat(),
               'directed': not self.is_symmetric()
              }
        prop.update(d)
        return prop


    def likelihood(self, theta, phi):
        likelihood = theta.dot(phi).dot(theta.T)
        return likelihood

    def template(self, d):
        d['time'] = d.get('time', None)
        netw_templ = '''###### $corpus
        Building: $time minutes
        Nodes: $instances
        Links: $nnz
        Degree mean: $nnz_mean
        Degree var: $nnz_var
        Diameter: $diameter
        Modularity: $modularity
        Clustering Coefficient: $clustering_coef
        Density: $density
        Communities: $communities
        Relations: $features
        Directed: $directed
        \n'''
        return super(frontendNetwork, self).template(d, netw_templ)

    def similarity_matrix(self, sim='cos'):
        features = self.features
        if features is None:
            return None

        if sim == 'dot':
            sim = np.dot(features, features.T)
        elif sim == 'cos':
            norm = np.linalg.norm(features, axis=1)[np.newaxis]
            sim = np.dot(features, features.T)/np.dot(norm.T, norm)
        elif sim == 'kmeans':
            cluster = kmeans(features, K=2)[np.newaxis]
            cluster[cluster == 0] = -1
            sim = np.dot(cluster.T,cluster)
        elif sim == 'comm':
            N = len(self.clusters)
            #sim = np.repeat(np.array(self.clusters)[np.newaxis], N, 0)
            theta , _ = self.get_params()
            sim = theta.dot(theta.T)
            sim = (sim == sim.T)*1
            sim[sim < 1] = -1
        elif sim == 'euclide_old':
            from sklearn.metrics.pairwise import euclidean_distances as ed
            #from plot import kmeans_plus
            #kmeans_plus(features, K=4)
            print (features)
            dist = ed(features)
            K = self.parameters_['k']
            devs = self.parameters_['devs'][0]
            sim = np.zeros(dist.shape)
            sim[dist <= 2.0 * devs / K] = 1
            sim[dist > 2.0  * devs / K] = -1
        elif sim == 'euclide_abs':
            from sklearn.metrics.pairwise import euclidean_distances as ed
            #from plot import kmeans_plus
            #kmeans_plus(features, K=4)
            N = len(features)
            K = self.parameters_['k']
            devs = self.parameters_['devs'][0]

            a = np.repeat(features[:,0][None], N, 0).T
            b = np.repeat(features[:,0][None], N, 0)
            sim1 = np.abs( a-b )
            a = np.repeat(features[:,1][None], N, 0).T
            b = np.repeat(features[:,1][None], N, 0)
            sim2 = np.abs( a-b )

            sim3 = np.zeros((N,N))
            sim3[sim1 <= 2.0*  devs / K] = 1
            sim3[sim1 > 2.0 *  devs / K] = -1
            sim4 = np.zeros((N,N))
            sim4[sim2 <= 2.0*  devs / K] = 1
            sim4[sim2 > 2.0 *  devs / K] = -1
            sim = sim4 + sim3
            sim[sim >= 0] = 1
            sim[sim < 0] = -1

        elif sim == 'euclide_dist':
            from sklearn.metrics.pairwise import euclidean_distances as ed
            #from plot import kmeans_plus
            #kmeans_plus(features, K=4)
            N = len(features)
            K = self.parameters_['k']
            devs = self.parameters_['devs'][0]

            sim1 = ed(np.repeat(features[:,0][None], 2, 0).T)
            sim2 = ed(np.repeat(features[:,0][None], 2, 0).T)

            sim3 = np.zeros((N,N))
            sim3[sim1 <= 2.0*  devs / K] = 1
            sim3[sim1 > 2.0 *  devs / K] = -1
            sim4 = np.zeros((N,N))
            sim4[sim2 <= 2.0*  devs / K] = 1
            sim4[sim2 > 2.0 *  devs / K] = -1
            sim = sim4 + sim3
            sim[sim >= 0] = 1
            sim[sim < 0] = -1
        return sim

    def homophily(self, model=None, sim='cos', type='kleinberg'):
        N = self.data.shape[0]
        card = N*(N-1)

        if model:
            data  = model.generate(N)
            #y = np.triu(y) + np.triu(y, 1).T
            gram_matrix = model.similarity_matrix(sim=sim)
            delta_treshold = .1
            gram_matrix[gram_matrix >= delta_treshold] = 1
            gram_matrix[gram_matrix < delta_treshold] = -1
        else:
            data = self.data
            gram_matrix = self.similarity_matrix(sim=sim)

        if gram_matrix is None:
            return np.nan, np.nan

        connected = data.sum()
        unconnected = card - connected
        similar = (gram_matrix > 0).sum()
        unsimilar = (gram_matrix <= 0).sum()

        indic_source = ma.array(np.ones(gram_matrix.shape)*-1, mask=ma.masked)
        indic_source[(data == 1) & (gram_matrix > 0)] = 0
        indic_source[(data == 1) & (gram_matrix <= 0)] = 1
        indic_source[(data == 0) & (gram_matrix > 0)] = 2
        indic_source[(data == 0) & (gram_matrix <= 0)] = 3

        np.fill_diagonal(indic_source, ma.masked)
        indic_source[indic_source == -1] = ma.masked

        a = (indic_source==0).sum()
        b = (indic_source==1).sum()
        c = (indic_source==2).sum()
        d = (indic_source==3).sum()

        if type == 'kleinberg':
            #print 'a: %s, connected: %s, similar %s, card: %s' % (a, connected,similar, card)
            homo_obs = 1.0 * a / connected # precision; homophily respected
            homo_exp = 1.0 * similar / card # rappel; strenght of homophily
        else:
            raise NotImplementedError

        #if sim == 'euclide' and type is None:
        #    homo_obs = 1.0 * (a + d - c - b) / card
        #    pr = 1.0 * (data == 1).sum() / card
        #    ps = 1.0 * (indic_source==0).sum() / card
        #    pnr = 1.0 - pr
        #    pns = 1.0 - ps
        #    a_ = pr*ps*card
        #    b_ = pnr*ps*card
        #    c_ = pr*pns*card
        #    d_ = pnr*pns*card
        #    homo_expect = (a_+b_-c_-d_) /card
        #    return homo_obs, homo_expect

        return homo_obs, homo_exp

    def assort(self, model):
        #if not source:
        #    data = self.data
        #    sim_source = self.similarity_matrix('cos')
        data = self.data
        N = self.data.shape[0]
        sim_source = self.similarity_matrix(sim='cos')

        y = model.generate(N)
        #y = np.triu(y) + np.triu(y, 1).T
        sim_learn = model.similarity_matrix(sim='cos')

        np.fill_diagonal(indic_source, ma.masked)

        assert(N == y.shape[0])

        indic_source = ma.array(np.ones(sim_source.shape)*-1, mask=ma.masked)
        indic_source[(data == 1) & (sim_source > 0)] = 0
        indic_source[(data == 1) & (sim_source <= 0)] = 1
        indic_source[(data == 0) & (sim_source > 0)] = 2
        indic_source[(data == 0) & (sim_source <= 0)] = 3

        indic_learn = ma.array(np.ones(sim_learn.shape)*-1, mask=ma.masked)
        indic_learn[(y == 1) & (sim_learn > 0)] = 0
        indic_learn[(y == 1) & (sim_learn <= 0)] = 1
        indic_learn[(y == 0) & (sim_learn > 0)] = 2
        indic_learn[(y == 0) & (sim_learn <= 0)] = 3

        np.fill_diagonal(indic_learn, ma.masked)
        np.fill_diagonal(indic_source, ma.masked)
        indic_source[indic_source == -1] = ma.masked
        indic_learn[indic_learn == -1] = ma.masked

        ### Indicateur Homophily Christine
        homo_ind1_source = 1.0 * ( (indic_source==0).sum()+(indic_source==3).sum()-(indic_source==1).sum() - (indic_source==2).sum() ) / (N*(N-1))
        homo_ind1_learn = 1.0 * ( (indic_learn== 0).sum()+(indic_learn==3).sum()-(indic_learn==1).sum() - (indic_learn==2).sum() ) / (N*(N-1))

        # AMI / NMI
        from sklearn import metrics
        AMI = metrics.adjusted_mutual_info_score(indic_source.compressed(), indic_learn.compressed())
        NMI = metrics.normalized_mutual_info_score(indic_source.compressed(), indic_learn.compressed())

        print('homo_ind1 source: %f' % (homo_ind1_source))
        print('homo_ind1 learn: %f' % (homo_ind1_learn))
        print('AMI: %f, NMI: %f' % (AMI, NMI))

        d = {'NMI' : NMI, 'homo_ind1_source' : homo_ind1_source, 'homo_ind1_learn' : homo_ind1_learn}
        return d

