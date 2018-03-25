import os
import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.sparse import lil_matrix, csr_matrix

from .frontend import DataBase
from .drivers import OnlineDatasetDriver

from pymake.util.math import *

try:
    import graph_tool as gt
    from graph_tool import collection as net_collection
    from graph_tool import stats, clustering, inference, spectral, topology, generation, search, draw
except Exception as e:
    print('Error while importing graph-tool: %s' % e)


class frontendNetwork_gt(DataBase, OnlineDatasetDriver):

    ''' A frontend layer for the graph-tool object.

        Attributes
        ----------
        data: Graph (from gt)

    '''

    def __init__(self, expe=None):
        super(frontendNetwork_gt, self).__init__(expe)
        self._data_type = 'network'

        data_format = expe.get('_data_format', 'b')
        if data_format in ['w', 'b']:
            self._net_type = data_format
        else:
            raise NotImplemented('Network format unknwown: %s' % data_format)


    @classmethod
    def _extract_data(cls, expe, corpus=None, remove_self_loops=True):
        corpus = corpus or {}
        input_path = expe._input_path
        data_format = expe.get('_data_format', 'b')
        force_directed = expe.get('directed', False)

        if expe.corpus.endswith('.gt'):
            corpus_name = expe.corpus[:-len('.gt')]
        else:
            corpus_name = expe.corpus

        # DB integration ?
        if corpus_name.startswith(('generator', 'graph')):
            ext = 'graph'
            basename = os.path.join(input_path, 't0.' + ext)
            fun = 'parse_dancer'
        elif corpus_name.startswith(('bench1')):
            raise NotImplementedError()
        elif corpus_name.startswith('facebook'):
            basename = 'edges'
            fn = os.path.join(input_path, '0.' + ext)
            fun = 'parse_edges'
        elif corpus_name.startswith(('manufacturing',)):
            ext = 'csv'
            basename = os.path.join(input_path, corpus_name +'.'+ ext)
            fun = 'parse_csv'
        elif corpus_name.startswith(('fb_uc', 'emaileu')):
            ext = 'txt'
            basename = os.path.join(input_path, corpus_name +'.'+ ext)
            fun = 'parse_tnet'
        elif corpus_name.startswith(('blogs','propro', 'euroroad')):
            ext = 'dat'
            basename = os.path.join(input_path, corpus_name +'.'+ ext)
            fun = 'parse_dat'
        else:
            raise ValueError('Which corpus to Load; %s ?' % corpus_name)

        fn = os.path.join(input_path, basename)
        if os.path.isfile(fn) and os.stat(fn).st_size == 0:
            cls.log.warning('Doh, Corpus file is empty at: %s' % fn)
            return

        parse_fun = getattr(cls, fun)

        directed = corpus.get('directed', force_directed)
        N = corpus.get('nodes')
        E = corpus.get('edges')

        g = gt.Graph(directed=directed)
        #g.add_vertex(N)

        labels = g.new_vertex_property("int") # g.new_vp
        weights = g.new_edge_property("int") # g.new_ep

        ### Slower but weight correct
        for obj in parse_fun(fn):

            n_nodes = g.num_vertices()

            if isinstance(obj, tuple):
                i, j, w, f = obj
                if max(i,j) >= n_nodes:
                    g.add_vertex(max(i,j) - n_nodes + 1)
                edge = g.edge(i,j)
                if not edge:
                    # keep a binary networks in the graph structure.
                    # Number of edge are in the {weights} property.
                    edge = g.add_edge(i, j)

                weights[edge] = w + weights[edge]
            else:
                v = obj.pop('index')
                if v >= n_nodes:
                    g.add_vertex(v - n_nodes + 1)
                labels[v] = obj['label']
        ### Faster but weigth wrong
        #_edges = []
        #_index = []
        #_label = []
        #for obj in parse_fun(fn):
        #    if isinstance(obj, tuple):
        #        i, j, w, f = obj
        #        _edges.append([i,j,w])
        #    else:
        #        _index.append(obj['index'])
        #        _label.append(obj['label'])

        #g.add_edge_list(_edges, eprops=[weights])
        #if _label:
        #    labels.a[_index] = _label

        g.vertex_properties['labels'] = labels # g.vp
        g.edge_properties['weights'] = weights # g.ep

        # Remove first index in case of indexation start at 1
        zero_degree = g.vertex(0).out_degree() + g.vertex(0).in_degree()
        if zero_degree == 0:
            g.remove_vertex(0)

        ## Remove selfloop
        if remove_self_loops:
            # @Warning: the corresponding weight are kept in the map properties
            cls.log.debug('Self-loop are assumed to be removde from the graph.')
            gt.stats.remove_self_loops(g)

        # If all lables are zeros, consider no label information
        if g.vp['labels'].a.sum() == 0:
            del g.vp['labels']

        _N = g.num_vertices()
        _E = g.num_edges()

        if N and N != _N:
            cls.log.warning('Number of nodes differs, doc: %d, code: %d' % (N, _N))
        if E and E != _E:
            cls.log.warning('Number of edges differs, doc: %d, code: %d' % (E, _E))

        return g


    @classmethod
    def _resolve_filename(cls, expe):
        input_path = expe._input_path

        if not os.path.exists(input_path):
            self.log.error("Corpus `%s' Not found." % (input_path))
            print('please run "fetch_networks"')
            self.data = None
            return

        if expe.corpus.endswith('.gt'):
            basename = expe.corpus
        else:
            basename = expe.corpus + '.gt'

        fn = os.path.join(input_path, basename)
        return fn

    @classmethod
    def _save_data(cls, fn, data):
        driver = data.save
        return super()._save_data(fn, None, driver=driver)

    @classmethod
    def _load_data(cls, fn):
        driver = gt.load_graph
        return super()._load_data(fn, driver=driver)


    @classmethod
    def from_expe(cls, expe, corpus=None, load=True, save=True):
        if '_force_load_data' in expe:
            load = expe._force_load_data
        if '_force_save_data' in expe:
            save = expe._force_save_data

        fn = cls._resolve_filename(expe)
        target_file_exists = os.path.exists(fn)

        if expe.corpus in net_collection.data:
            data = gt.collection.data[expe.corpus]
        elif load is False or not target_file_exists:
            data = cls._extract_data(expe, corpus=corpus)
            if save:
                # ===== save ====
                cls._save_data(fn, data)
        else:
            # ===== load ====
            data = cls._load_data(fn)

        frontend = cls(expe)
        frontend.data = data

        #exit()
        return frontend



    #
    # Get Statistics
    #

    def is_symmetric(self):
        return not self.data.is_directed()

    def num_nodes(self):
        return self.data.num_vertices()

    def num_edges(self):
        return self.data.num_edges()

    def diameter(self):
        diameter, end_points = gt.topology.pseudo_diameter(self.data)
        return int(diameter)

    def density(self):
        e = self.data.num_edges()
        v = self.data.num_vertices()
        t = v * (v-1)
        if not self.data.is_directed():
            t /= 2
        return e / t

    def modularity(self):
        if not 'labels' in self.data.vp:
            return None
        labels = self.data.vp['labels']

        if self._net_type == 'w':
            weights = self.data.ep['weights']
        else:
            weights = None

        mod = gt.inference.modularity(self.data, labels, weights)
        return mod

    def clustering_coefficient(self):
        clust_coef, std = gt.clustering.global_clustering(self.data)
        return clust_coef

    def net_type(self):
        weights = self.data.ep['weights'].a
        return '%s / min-max: %s %s' % (self._net_type,
                                               weights.min(),
                                               weights.max())

    def feat_len(self):
        weights = self.data.ep['weights'].a
        return len(np.unique(weights))

    @staticmethod
    def _prop(g):
        n, e, y = g.num_vertices(), g.num_edges(), gt.spectral.adjacency(g)
        sum = y.sum()
        sl = np.diag(y.A).sum()
        print(g)
        print('N: %d, E: %d, adj.sum: %d, adj.selfloop: %d' % (n,e,sum,sl))
        print('edges shape', g.get_edges().shape)


    #
    # Transform
    #

    def adj(self):
        ''' Returne a sparse Adjacency matrix. '''
        return gt.spectral.adjacency(self.data)

    def laplacian(self):
        ''' Returns a sparse Laplacian matrix. '''
        return gt.spectral.laplacian(self.data)


    def binarize(self):
        # Or set a propertie with gt.stats.label_parallel_edges(g)
        return gt.stats.remove_parallel_edges(self.data)

    def remove_self_loops(self):
        # Or set a propertie with gt.stats.remove_self_loops(g)
        return gt.stats.remove_self_loops(self.data)

    def sample(self, N):
        ''' Reduce the numbre of nodes of the graph. '''
        n_to_remove = self.data.num_vertices() - N
        ids = np.random.randint(0, self.data.num_vertices(), n_to_remove)
        self.data.remove_vertex(ids)




    #
    # Test/Validation set
    #

    def make_testset(self, percent_hole, diag_off=1):
        ''' make the testset as a edge propertie of the graph. '''

        percent_hole = float(percent_hole)
        if percent_hole >= 1:
            percent_hole = percent_hole / 100
        elif 0 <= percent_hole < 1:
            pass
        else:
            raise ValueError('cross validation ratio not understood : %s' % percent_hole)

        mask_type =  self.expe.get('mask', 'unbalanced')
        #if mask_type == 'unbalanced':
        #    testset = self.get_masked(percent_hole, diag_off)
        #elif mask_type == 'balanced':
        #    testset = self.get_masked_balanced(percent_hole, diag_off)
        #elif mask_type == 'zeros':
        #    testset = self.get_masked_zeros(diag_off)
        #else:
        #    raise ValueError('mask type unknow :%s' % mask_type)

        g = self.data
        adj = self.adj()
        N = g.num_vertices()
        E = g.num_edges()
        if self.is_symmetric():
            T = N*(N-1)/2
        else:
            T = N*(N-1)

        #validset = lil_matrix((N,N))
        testset = lil_matrix((N,N))
        n = int(E * percent_hole)

        # Sampling non-zeros
        edges = g.get_edges().astype(int)
        #i,j = adj.nonzero()
        i, j = edges[:, 0:2].T
        ix = np.random.choice(E, n, replace=False)
        ix = (i[ix], j[ix])
        testset[ix[0], ix[1]] = 1


        cpt=0
        cc=0
        for e in edges:
            f = adj[e[0], e[1]]
            if f == 0:
                cpt +=1
                print(g.ep['weights'][e[0],e[1]])
                print(g.edge(e[0], e[1]), g.edge(e[1], e[0]))
            else:
                print(g.edge(e[0], e[1]), g.edge(e[1], e[0]))
                cc +=1
        print('non edges !!!', cpt)
        print('edges !!!', cc)
        print('totlat', cc+cpt)
        print(g.is_directed())
        print(N, E, adj.shape)
        # Now, Ignore the diagonal => include the diag indices
        # in (i,j) like if it they was selected in the testset.
        size = T + N
        i = np.hstack((i, np.arange(N)))
        j = np.hstack((j, np.arange(N)))


        # Sampling zeros
        k = np.ravel_multi_index((i,j), adj.shape)
        jx = np.random.choice(size, n, replace=False)

        kind = np.bincount(k, minlength=size).astype(bool)
        jind = np.bincount(jx, minlength=size).astype(bool)

        # xor on bool fastar ?!
        _n = (jind & (kind==0)).sum()
        max_loop = 100
        cpt = 0
        while _n < n or cpt > max_loop:
            jx = np.random.choice(T, n - _n, replace=False)
            jind = jind|np.bincount(jx, minlength=size).astype(bool)
            _n = (jind & (kind==0)).sum()
            cpt += 1
            print(_n, n)

        jx = np.arange(size)[jind & (kind==0)]
        jx = np.unravel_index(jx, (N,N))
        testset[jx[0], jx[1]] = 1

        self._links_testset = ix
        self._nonlinks_testset = jx


        return testset
