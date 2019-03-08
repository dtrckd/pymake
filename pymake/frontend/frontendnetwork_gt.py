import os
import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.sparse import lil_matrix, csr_matrix

from .frontend import DataBase
from .drivers import OnlineDatasetDriver

try:
    import graph_tool as gt
    from graph_tool import stats, clustering, inference, spectral, topology, generation, search
    from graph_tool import draw
except Exception as e:
    print('Error while importing graph-tool: %s' % e)


class frontendNetwork_gt(DataBase, OnlineDatasetDriver):

    ''' A frontend layer for the graph-tool object.

        Attributes
        ----------
        data: Graph (from gt)

    '''

    def __init__(self, expe, data, corpus=None):
        super().__init__(expe)
        self._data_type = 'network'
        corpus = corpus or {}

        self.data = data

        force_directed = expe.get('directed', corpus.get('directed'))
        force_directed = bool(force_directed) if force_directed is not None else None
        remove_self_loops = expe.get('remove_self_loops', True)

        # Remove selfloop
        if remove_self_loops:
            # @Warning: the corresponding weight are kept in the map properties
            self.log.debug('Self-loop are assumed to be remove from the graph.')
            self.remove_self_loops()

        # Set the direction and weights
        if force_directed is not None:
            self.set_directed(force_directed)
        else:
            # Try to guess directed or not
            y = self.adj()
            adj_symmetric = (y!=y.T).nnz == 0
            adj_symmetric |= sp.sparse.triu(y,1).nnz == 0
            adj_symmetric |= sp.sparse.tril(y,-1).nnz == 0
            if adj_symmetric:
                self.log.info('Guessing undirected graph for: %s' % expe.corpus)
                self.set_directed(False)

            if y.max() > 1:
                self.log.critical('Multiple edges in the graph..')

        if not 'weights' in data.ep:
            if 'weight' in data.ep:
                self.log.critical('Already Weight propertye in the graph, need to check ')
                raise NotImplementedError
            elif 'value' in data.ep:
                weights = data.ep['value'].copy()
                #w.a = (2**0.5)**w.a # exponentiate
                #weights.a = np.round((weights.a+1)**2) # squared
                weights.a = np.round(weights.a *10) # assume ten authors by paper for (collaboration networks)
                data.ep['weights'] = weights.copy('int')
            else:
                weights = data.new_ep("int")
                weights.a = 1
                data.ep['weights'] = weights
        else:
            weights = data.ep.weights

        if self.expe.get('shift_w'):
            shift = self.expe.get('shift_w')
            if str(shift).startswith('linear'):
                _, shift = shift.split('_')
                #shift = weights.a.max()
                shift = int(shift)
                expe['shift_w'] = shift
                weights.a = shift*weights.a +shift
            else:
                shift = int(self.expe['shift_w'])
                weights.a += shift



    @classmethod
    def _extract_data_file(cls, expe, corpus=None):
        corpus = corpus or {}
        input_path = expe._input_path

        if not os.path.exists(input_path):
            cls.log.error("Corpus `%s' Not found." % (input_path))
            print('please run "fetch_networks"')
            cls.data = None
            return

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

        N = corpus.get('nodes')
        E = corpus.get('edges')

        g = gt.Graph(directed=True)
        #g.add_vertex(N)

        weights = g.new_edge_property("int") # g.new_ep
        labels = g.new_vertex_property("string") # g.new_vp
        clusters = g.new_vertex_property("int") # g.new_vp

        ### Slower but weight correct
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

                if 'label' in obj:
                    labels[v] = obj['label']
                if 'cluster' in obj:
                    clusters[v] = obj['cluster']

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

        g.edge_properties['weights'] = weights # g.ep
        g.vertex_properties['labels'] = labels # g.vp
        g.vertex_properties['clusters'] = clusters

        # If nolabels remove that property
        n_label = 0
        for v in range(g.num_vertices()):
            if g.vp['labels']:
                n_label += 1
                break

        if n_label == 0:
            del g.vp['labels']
        # If all clusters are zeros, consider no label information
        if g.vp['clusters'].a.sum() == 0:
            del g.vp['clusters']

        # Remove first index in case of indexation start at 1
        zero_degree = g.vertex(0).out_degree() + g.vertex(0).in_degree()
        if zero_degree == 0:
            cls.log.debug('shifting the graph. (-1 to vertex index)')
            g.remove_vertex(0)

        _N = g.num_vertices()
        _E = g.num_edges()

        if N and N != _N:
            cls.log.warning('Number of nodes differs, doc: %d, code: %d' % (N, _N))
        if E and E != _E:
            cls.log.warning('Number of edges differs, doc: %d, code: %d' % (E, _E))

        return g


    @classmethod
    def _clean_data_konect(cls, expe, data):
        cls.log.info('Cleaning konect graph %s...' % expe.corpus)

        g = data
        edges = g.get_edges()
        y = gt.spectral.adjacency(g)
        parallel = gt.stats.label_parallel_edges(g, mark_only=True)

        weights = g.new_ep('int')

        if 'weight' in g.ep:

            w = g.ep.weight

            if w.a.sum() == len(w.a):
                # Assume parrallel edge
                # Building the real weights
                weights.a = 1
                for _ix in np.where(parallel.a == 0)[0]:
                    ix = np.where(edges[:,2]==_ix)[0][0]
                    i, j = edges[ix, :2]
                    weights[i,j] += len(g.edge(i,j, all_edges=True))


                del g.ep['weight']
            elif w.a.max() == 1 and w.a.min() == -1:
                for i,j, ix in edges:
                    weights[i,j] += w[i,j]

                if weights.a.min() < 0:
                    # Remove negative edges
                    cls.log.warning('Negative weights here? removing edges...')
                    print('Number of negative edges: %d' % (weights.a < 0).sum() )
                    #print('Number of negative edges: %d' % (weights.a < 0).sum() )
                    #edge_f = g.new_ep('bool')
                    #edge_f.a = 1
                    #edge_f.a[weights.a < 0] = 0
                    #g.set_edge_filter(edge_f)
                    #g.purge_edges()
                    #g.clear_filters()
                    weights.a[weights.a < 0] = 1

                if (weights.a==0).sum() > 0:
                    # Remove empty edges
                    cls.log.warning('Removing zeros weighted edges...')
                    print('Number of zeros edges: %d' % (weights.a == 0).sum() )
                    #edge_f = g.new_ep('bool')
                    #edge_f.a = 1
                    #edge_f.a[weights.a == 0] = 0
                    #g.set_edge_filter(edge_f)
                    #g.purge_edges()
                    #g.clear_filters()
                    weights.a[weights.a == 0] = 1

            elif y.max() == 1:
                # Assume weight are the real one, copy
                weights = g.ep.weight.copy('int')
                cls.log.warning('Weight regression to int values.')

            elif y.max() > 1:
                raise NotImplementedError('Multi edge + weight property for %s, manual check needed' % expe.corpus)
            else:
                raise NotImplementedError('Edge propertie unknown for %s, manual check needed' % expe.corpus)

        elif 'weights' in g.ep:
            self.log.critical('Already weights property for %s, manual check needed' % expe.corpus)
            raise NotImplementedError
        elif y.max() > 1:
            # If multiple edge detected, set the weights.
            weights.a = 1
            for _ix in np.where(parallel.a == 0)[0]:
                ix = np.where(edges[:,2]==_ix)[0][0]
                i, j = edges[ix][:2]
                weights[i,j] += len(g.edge(i,j, all_edges=True))
        else:
            # Assume unweighted network
            weights.a = 1


        # Remove parallel edges
        #gt.stats.remove_parallel_edges(g) # dont't resize .a !
        g.set_edge_filter(parallel, inverted=True)
        g.purge_edges()
        g.clear_filters()

        g.shrink_to_fit()
        g.ep['weights'] = weights
        return g


    @classmethod
    def _resolve_filename(cls, expe):
        input_path = expe._input_path

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

        input_path = cls.get_input_path(expe)

        data = None
        fn = cls._resolve_filename(expe)
        target_file_exists = os.path.exists(fn)

        if load is False or not target_file_exists:

            # Data loading Strategy
            if not data:
                try:
                    # Load from graph-tool Konnect repo
                    from graph_tool import collection
                    data = gt.load_graph(collection.get_data_path(expe.corpus))
                    os.makedirs(os.path.join(input_path), exist_ok=True)
                except FileNotFoundError as e:
                    pass
                except Exception as e:
                    cls.log.error("Error in loading corpus `%s': %s" % (expe.corpus, e))
                    raise e

            if not data:
                try:
                    from urllib.error import HTTPError
                    # Load from graph-tool Konnect site
                    data = gt.collection.konect_data[expe.corpus]
                    data = cls._clean_data_konect(expe, data)
                    os.makedirs(os.path.join(input_path), exist_ok=True)
                except HTTPError:
                    pass
                except Exception as e:
                    cls.log.error("Error in loading corpus `%s': %s" % (expe.corpus, e))
                    raise e

            if not data:
                # Load manually from file
                data = cls._extract_data_file(expe, corpus=corpus)

            if save:
                # ===== save ====
                cls._save_data(fn, data)
        else:
            # ===== load ====
            data = cls._load_data(fn)

        return cls(expe, data, corpus=corpus)


    #
    # Get Properties
    #

    def edge(self, i,j):
        return self.data.edge(i,j)

    def weight(self, i,j):
        w = self.data.ep['weights']
        if self.edge(i,j):
            return w[i,j]
        else:
            return 0

    def label(self, v):
        l = self.data.vp['labels']
        return l[v]

    def cluster(self, v):
        c = self.data.vp['clusters']
        return c[v]

    def num_neighbors(self):
        neigs = []
        for v in range(self.num_nodes()):
            _v = self.data.vertex(v)
            neigs.append(len(list(_v.all_neighbors())))
        return np.asarray(neigs)

    def is_directed(self):
        return self.data.is_directed()
    def is_symmetric(self):
        return not self.data.is_directed()

    def getN(self):
        return self.num_nodes()

    def get_validset_ratio(self):

        if 'validset_ratio' in self.expe:
            validset_ratio = self.expe['validset_ratio']
            validset_ratio = float(validset_ratio) / 100
        else:
            validset_ratio = 0

        return validset_ratio

    def get_training_ratio(self):

        if 'training_ratio' in self.expe:
            training_ratio = self.expe['training_ratio']
            training_ratio = float(training_ratio) / 100
        else:
            training_ratio = 0

        return training_ratio

    def get_testset_ratio(self):

        if 'testset_ratio' in self.expe:
            testset_ratio = self.expe['testset_ratio']
            testset_ratio = float(testset_ratio) / 100
        else:
            testset_ratio = 0

        validset_ratio = self.get_validset_ratio()

        # Validation ratio is a ratio took on the remaining data (eta_ratio...)
        testset_ratio = testset_ratio * (1 + validset_ratio)

        return testset_ratio

    def num_nodes(self):
        return self.data.num_vertices()

    def num_edges(self):
        return self.data.num_edges()

    def num_nnz(self):
        N = self.data.num_vertices()
        sym_pt = 1 if self.is_directed() else 2
        if hasattr(self, 'data_test'):
            T = N*(N-1)/sym_pt - int(self.data_test.sum()/sym_pt)
        else:
            T = N*(N-1)/sym_pt

        return T

    def num_nnzsum(self):
        return self.data.ep['weights'].a.sum()

    def num_mnb(self):
        ''' Minibatche size for the sampling. '''
        return int(self.num_nodes() * self._zeros_set_len * float(self.expe['sampling_coverage']))

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

        if 'weights' in self.data.ep:
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
        return 'm/std/mn/mx: %.1f %.1f %d %d' % (weights.mean(),
                                                      weights.std(),
                                                      weights.min(),
                                                      weights.max())

    def feat_len(self):
        weights = self.data.ep['weights'].a
        return len(np.unique(weights))

    @staticmethod
    def _prop(g):
        n, e, y = g.num_vertices(), g.num_edges(), gt.spectral.adjacency(g)
        sum = y.sum()
        #sl = np.diag(y.A).sum()
        sl = y[(np.arange(n),np.arange(n))].sum()
        print(g)
        print('N: %d, E: %d, adj.sum: %d, adj.selfloop: %d' % (n,e,sum,sl))
        print('edges shape', g.get_edges().shape)
        print('Vertex prop', g.vp)
        print('Edge prop', g.ep)


    #
    # Transform
    #

    def reverse_filter(self):
        filter, is_inv = self.data.get_edge_filter()
        self.data.set_edge_filter(filter, inverted=not is_inv)

    def adj(self):
        ''' Returns a sparse Adjacency matrix.

            Notes
            -----
            index of the adjacency are reversed from np convention i:line, j:column
        '''
        return gt.spectral.adjacency(self.data).T

    def laplacian(self):
        ''' Returns a sparse Laplacian matrix. '''
        return gt.spectral.laplacian(self.data)


    def binarize(self):
        # Or set a propertie with gt.stats.label_parallel_edges(g)
        return gt.stats.remove_parallel_edges(self.data)

    def set_directed(self, directed=True):
        # Modifying the adjency matrix don't modify the gt graph
        #y = self.adj()
        #(y!=y.T).nnz == 0
        self.data.set_directed(directed)

    def remove_self_loops(self):
        # Or set a propertie with gt.stats.remove_self_loops(g)
        return gt.stats.remove_self_loops(self.data)

    def sample(self, N):
        ''' Reduce randomly  the number of nodes of the graph. '''
        n_to_remove = self.data.num_vertices() - N
        ids = np.random.randint(0, self.data.num_vertices(), n_to_remove)
        self.data.remove_vertex(ids)
        self.data.shrink_to_fit()



    #
    # Test/Validation set
    #

    def make_testset(self, diag_off=1, set_filter=True):
        ''' make the testset as a edge propertie of the graph.
            If {set_filter} is True, activate the filtering property on the graph

            The algo try to build a equillibrite testset between edge and non-edge with
            some uncertanty for undirected graph.

            Notes
            -----
            For Undirected graph, the sparse matrice is symmetrized, so the value of
            edges can be cummulated from the two inverted sample. Thus the ratio for the
            non-edges is reduce to 1/3 of the edge ratio to be fair.

            This operation can be sub-optimal since it compare array of size N**2.

            Warning: if set_filter=false, wiehgt is not set in the testset matrix.
        '''

        testset_ratio = self.get_testset_ratio()
        if not testset_ratio:
            raise ValueError('Testset ratio not understood : %s' % testset_ratio)

        g = self.data
        y = self.adj()
        N = g.num_vertices()
        E = g.num_edges()
        is_directed = not self.is_symmetric()
        symmetric_scale = 1 if is_directed else 2
        size = N**2

        #validset = lil_matrix((N,N))
        testset = lil_matrix((N,N), dtype=bool)
        n = int(E * testset_ratio) # number of edge
        nz = int(n *float(self.expe.get('zeros_ratio', 1)))  # number of non-links

        edges = g.get_edges().astype(int)
        i, j = edges[:, 0:2].T

        # Sampling edges
        ix = np.random.choice(E, n, replace=False)
        ix = np.array((i[ix], j[ix]))
        #testset[ix[0], ix[1]] = 1

        # Sampling non-edges
        #
        # Ignore the diagonal => include the diag indices
        # in (i,j) like if it they was selected in the testset.
        # If Undirected also force ignore the lower part of the adjacency matrix.
        if is_directed:
            i = np.hstack((i, np.arange(N)))
            j = np.hstack((j, np.arange(N)))
        else:
            _i = i
            i = np.hstack((i, j, np.arange(N)))
            j = np.hstack((j, _i, np.arange(N)))

        edge_set = set(np.ravel_multi_index((i,j), y.shape))

        nonlinks_set = set()
        max_loop = 150
        cpt = 0
        # Add the probability to sample symmetric edge...
        zeros_len = int(nz + int(not is_directed)*nz/size)
        while len(nonlinks_set) < zeros_len or cpt >= max_loop:
            nnz = zeros_len - len(nonlinks_set)
            nonlinks_set.update(set(np.random.choice(size, nnz)))
            nonlinks_set -= edge_set
            cpt+=1

        self.log.debug('Number of iteration for sampling non-edges: %d' % cpt)
        if cpt >= max_loop:
            self.log.warning('Sampling tesset nonlinks failed:')
            print('desires size: %d, obtained: %s' % (zeros_len, len(nonlinks_set)))

        jx = np.array(np.unravel_index(list(nonlinks_set), (N,N)))

        #k = np.ravel_multi_index((i,j), y.shape)
        #jx = np.random.choice(size, int(nz/symmetric_scale), replace=False)

        #kind = np.bincount(k, minlength=size).astype(bool)
        #jind = np.bincount(jx, minlength=size).astype(bool)

        ## xor on bool faster ?!
        #_n = (jind & (kind==0)).sum()
        #max_loop = 100
        #cpt = 0
        #while _n < int(nz/symmetric_scale) or cpt > max_loop:
        #    jx = np.random.choice(size, int(nz/symmetric_scale) - _n, replace=False)
        #    jind = jind|np.bincount(jx, minlength=size).astype(bool)
        #    _n = (jind & (kind==0)).sum()
        #    cpt += 1
        #    #print(_n, n)
        #self.log.debug('Number of iteration for sampling non-edges: %d'%cpt)

        #jx = np.arange(size)[jind & (kind==0)]
        #jx = np.array(np.unravel_index(jx, (N,N)))


        # Actually settings the testset entries
        testset[ix[0], ix[1]] = 1
        testset[jx[0], jx[1]] = 1
        if not is_directed:
            testset[ix[1], ix[0]] = 1
            testset[jx[1], jx[0]] = 1


        data_test_w = np.transpose(testset.nonzero())
        #self.reverse_filter()
        weights = []
        for i,j in data_test_w:
            weights.append(self.weight(i,j))
        #self.reverse_filter()
        data_test_w = np.hstack((data_test_w, np.array(weights)[None].T))

        if set_filter:
            # Also set the weight here...
            testset_filter = g.new_ep('bool')
            testset_filter.a = 1
            self.log.info('Looping to set the testset edge eproperties (graph-tool)...')
            for i, j in ix.T:
                testset_filter[i,j] = 0
            g.set_edge_filter(testset_filter)

        self._links_testset = ix
        self._nonlinks_testset = jx
        self.data_test = testset
        self.data_test_w = data_test_w

        # if we learn with a training_ratio < 1
        # we just ignore some random point.
        training_ratio = self.get_training_ratio()
        if training_ratio > 0 and training_ratio < 1:
            # remove edges
            self.data.set_fast_edge_removal()

            edges = self.data.get_edges()
            weights = self.data.ep['weights']
            n_extra = len(edges) - training_ratio*len(edges)

            edges_to_remove = np.random.choice(len(edges), int(n_extra), replace=False)
            for pos in edges_to_remove:
                i,j = edges[pos, :2]
                e = g.edge(i,j)
                g.remove_edge(e)

            self.data.set_fast_edge_removal(fast=False)
        return

    def _check(self):
        ''' Warning: will clear the filter! '''
        expe = self.expe
        g = self.data
        self._prop(g)

        y = self.adj()
        N = g.num_vertices()
        E = g.num_edges()

        # check no self-loop
        assert(y.diagonal().sum() == 0)
        # check directed/undirected is consistent
        is_directed = not self.is_symmetric()
        adj_symmetric = (y!=y.T).nnz == 0
        assert(adj_symmetric != is_directed)
        # check is not bipartite
        assert(not gt.topology.is_bipartite(g))

        if 'testset_ratio' in expe:
            testset_ratio = self.get_testset_ratio()
            filter = g.get_edge_filter()
            g.clear_filters()
            y = self.adj()
            n = int(g.num_edges() * testset_ratio)
            nz = int(n *float(self.expe.get('zeros_ratio', 1)))
            ix = self._links_testset
            jx = self._nonlinks_testset
            # check links testset
            assert(np.all(y[ix[0], ix[1]] == 1))
            # check non-links testset
            assert(np.all(y[jx[0], jx[1]] == 0))

            print('Size of testset: expected: %d, obtained: %d' % (n+nz, self.data_test.sum()))
            print('number of links: %d' % (len(ix[0])))
            print('number of non-links: %d' % (len(jx[0])))
            if filter[0]:
                g.set_edge_filter(filter[0], inverted=filter[1])

        return

    def make_noise(self):
        expe = self.expe
        g = self.data
        is_directed = g.is_directed()

        symmetric_scale = 1 if is_directed else 2

        y = self.adj()
        E = g.num_edges()
        N = g.num_vertices()
        T = N*(N-1)/symmetric_scale

        weights = g.ep['weights']

        ratio = 1/float(expe['noise'])
        #ts = self.get_testset_ratio()
        #if ts:
        #    ratio -= 2*ts*E/T
        #    if ratio < 0:
        #        raise ValueError('Negative prior for noise ration')
        g.set_fast_edge_removal()

        edges = g.get_edges()
        edges_to_remove = np.random.choice(E, int(ratio*E), replace=False)
        _removed = set()
        for pos in edges_to_remove:
            i,j = edges[pos, :2]
            w = weights[i,j]
            if w > 1:
                weights[i,j] = w-1
            else:
                e = g.edge(i,j)
                _removed.add(e)
                g.remove_edge(e)

        num_to_add = ratio*E*2
        cpt = 0
        while cpt < num_to_add:
            i, j = np.random.randint(0,N, 2)
            if g.edge(i,j):
                continue
            else:
                e = g.add_edge(i,j)
                if e in _removed:
                    continue
                weights[e] = 1
                cpt+=1

        return g.set_fast_edge_removal(fast=False)

    def __iter__(self):

        # Get the sampling strategy:
        chunk = self.expe.get('chunk', 'stratify')

        if chunk == 'stratify':
            return self._sample_stratify()
        elif chunk == 'sparse':
            return self._sample_sparse()
        else:
            raise NotImplementedError('Unkonw sampling strategy: %s' % chunk)


    def _sample_stratify(self):
        ''' Sample with node replacement.
            But edges in a minibatch is unique.

            Returns
            ------

            yield ether: a triplet (source, target, weight), that can be of two kinf
            1. at a new minibatch (mnb):
                * source: str -- a str that identify from which subset the mnb comes.
                * target: int -- the target node of the mnb.
                * weight: int -- the probability of to sample a element in the current set/mnb
                          in the corpus/dataset.
            2. an obeservation (edges):
                * source: int -- source node.
                * target: int -- target node.
                * weight: int -- edge weight.
        '''
        expe = self.expe
        g = self.data
        is_directed = g.is_directed()

        symmetric_scale = 1 if is_directed else 2 # number of set containg each edges

        y = self.adj()
        E = g.num_edges()
        N = g.num_vertices()
        T = N*(N-1)/symmetric_scale

        weights = g.ep['weights']

        ### Chunk prior probability
        set_len = 2 + is_directed
        # uniform between links and non-links
        zero_prior = float(expe.get('zeros_set_prob', 0.5))
        set_prior = [zero_prior] + [(1-zero_prior)/(set_len-1)]*(set_len-1)

        ### Sampling prior strategy
        zeros_set_len = expe.get('zeros_set_len', 10)
        self._zeros_set_len = int(zeros_set_len)
        mask = np.ones((N,2), dtype=int) * self._zeros_set_len

        ### Build the array of neighbourhood for each nodes
        self.log.debug('Building the neighbourhood array...')
        neigs = []
        mask_pos = []
        for v in range(N):
            _out = np.array([int(_v) for _v in g.vertex(v).out_neighbors()])
            _in = np.array([int(_v) for _v in g.vertex(v).in_neighbors()])
            neigs.append([_out, _in])
            mask_pos.append( [list(range(self._zeros_set_len))]*2 )


        for _mnb in range(self.num_mnb()):

            # Pick a node
            node = np.random.randint(0,N)

            # Pick a set and yield a minibatch
            set_index = np.random.choice(set_len, 1, p=set_prior)

            if set_index == 0:
                out_e = neigs[node][0]
                if N-len(out_e) > 0 and len(out_e) > 0:

                    node_info = {'vertex':node, 'direction':0}
                    yield str(set_index), node_info, 1/symmetric_scale*set_len*mask[node, 0]
                    # Sample from non-links

                    zero_samples = np.arange(N)

                    zero_samples[out_e] = -1 # don't sample in edge
                    zero_samples[node] = -1 # don't sample in self loops
                    #zero_samples[self.data_test[node, :].nonzero()[1]] = -1 # don't sample in testset
                    zero_samples = zero_samples[zero_samples >0]

                    zero_samples = np.random.choice(zero_samples,
                                                     int(np.ceil(len(zero_samples)/mask[node, 0])),
                                                     replace=False)
                    # Weaker results !
                    #step = int(np.ceil(len(zero_samples)/mask[node, 0]))
                    #if len(mask_pos[node][0]) == 0:
                    #    continue
                    #    zero_samples = np.random.choice(zero_samples,
                    #                                    int(np.ceil(len(zero_samples)/mask[node, 0])),
                    #                                    replace=False)
                    #else:
                    #    _p = np.random.choice(mask_pos[node][0],1).item()
                    #    mask_pos[node][0].remove(_p)
                    #    zero_samples = zero_samples[_p:_p+step]

                    if len(zero_samples) == 0:
                        continue

                    for target in zero_samples:
                        yield node, target, 0

                in_e = neigs[node][1]
                if N-len(in_e) > 0 and len(in_e) > 0:
                    node_info = {'vertex':node, 'direction':1}
                    yield str(set_index), node_info, 1/symmetric_scale*set_len*mask[node, 1]

                    zero_samples = np.arange(N)
                    if len(in_e) > 0:
                        zero_samples[in_e] = -1
                    zero_samples[node] = -1
                    #zero_samples[self.data_test[:, node].nonzero()[0]] = -1
                    zero_samples = zero_samples[zero_samples >0]

                    zero_samples = np.random.choice(zero_samples,
                                                    int(np.ceil(len(zero_samples)/mask[node, 1])),
                                                    replace=False)
                    # Weaker results !
                    #step = int(np.ceil(len(zero_samples)/mask[node, 1]))
                    #if len(mask_pos[node][1]) == 0:
                    #    continue
                    #    zero_samples = np.random.choice(zero_samples,
                    #                                    int(np.ceil(len(zero_samples)/mask[node, 1])),
                    #                                    replace=False)
                    #else:
                    #    _p = np.random.choice(mask_pos[node][1],1).item()
                    #    mask_pos[node][1].remove(_p)
                    #    zero_samples = zero_samples[_p:_p+step]

                    if len(zero_samples) == 0:
                        continue

                    for target in zero_samples:
                        yield target, node, 0

            elif set_index == 1:
                # Sample from links
                if len(neigs[node][0]) == 0:
                    continue

                node_info = {'vertex':node, 'direction':0}
                yield str(set_index), node_info, 1/symmetric_scale*N*set_len

                for target in neigs[node][0]:
                    yield node, target, weights[node, target]
            elif set_index == 2:
                # Sample from links
                if len(neigs[node][1]) == 0:
                    continue

                node_info = {'vertex':node, 'direction':1}
                yield str(set_index), node_info, 1/symmetric_scale*N*set_len

                for target in neigs[node][1]:
                    yield target, node, weights[target, node]
            else:
                raise ValueError('Set index error: %s' % set_index)


    def _sample_sparse(self):
        ''' Sample with node replacement.
        '''
        expe = self.expe
        g = self.data
        is_directed = g.is_directed()

        symmetric_scale = 1 if is_directed else 2 # number of set containg each edges

        y = self.adj()
        E = g.num_edges()
        N = g.num_vertices()
        T = N*(N-1)/symmetric_scale

        weights = g.ep['weights']

        ### Chunk prior probability
        set_len = 2 + is_directed
        # uniform between links and non-links
        zero_prior = float(expe.get('zeros_set_prob', 0.5))
        set_prior = [zero_prior] + [(1-zero_prior)/(set_len-1)]*(set_len-1)

        ### Sampling prior strategy
        zeros_set_len = expe.get('zeros_set_len', 10)
        self._zeros_set_len = int(zeros_set_len)
        mask = np.ones((N,2), dtype=int) * self._zeros_set_len

        ### Build the array of neighbourhood for each nodes
        self.log.debug('Building the neighbourhood array...')
        neigs = []
        mask_pos = []
        for v in range(N):
            _out = np.array([int(_v) for _v in g.vertex(v).out_neighbors()])
            _in = np.array([int(_v) for _v in g.vertex(v).in_neighbors()])
            neigs.append([_out, _in])
            mask_pos.append( [list(range(self._zeros_set_len))]*2 )


        for _mnb in range(self.num_mnb()):

            # Pick a node
            node = np.random.randint(0,N)

            # Pick a set and yield a minibatch
            set_index = np.random.choice(set_len, 1, p=set_prior)

            if set_index == 0:
                out_e = neigs[node][0]
                if N-len(out_e) > 0 and len(out_e) > 0:

                    node_info = {'vertex':node, 'direction':0}
                    yield str(set_index), node_info, (N-len(out_e)-1)* mask[node,0]* symmetric_scale
                    # Sample from non-links

                    zero_samples = np.arange(N)

                    zero_samples[out_e] = -1 # don't sample in edge
                    zero_samples[node] = -1 # don't sample in self loops
                    #zero_samples[self.data_test[node, :].nonzero()[1]] = -1 # don't sample in testset
                    zero_samples = zero_samples[zero_samples >0]
                    zero_samples = np.random.choice(zero_samples,
                                                     int(np.ceil(len(zero_samples)/mask[node, 0])),
                                                     replace=False)
                    # Weaker results !
                    #step = int(np.ceil(len(zero_samples)/mask[node, 0]))
                    #if len(mask_pos[node][0]) == 0:
                    #    zero_samples = np.random.choice(zero_samples,
                    #                                    int(np.ceil(len(zero_samples)/mask[node, 0])),
                    #                                    replace=False)
                    #else:
                    #    _p = np.random.choice(mask_pos[node][0],1).item()
                    #    mask_pos[node][0].remove(_p)
                    #    zero_samples = zero_samples[_p:_p+step]

                    if len(zero_samples) == 0:
                        continue

                    for target in zero_samples:
                        yield node, target, 0

                in_e = neigs[node][1]
                if N-len(in_e) > 0 and len(in_e) > 0:
                    node_info = {'vertex':node, 'direction':1}
                    yield str(set_index), node_info, (N-len(in_e)-1)* mask[node,1]* symmetric_scale

                    zero_samples = np.arange(N)
                    if len(in_e) > 0:
                        zero_samples[in_e] = -1
                    zero_samples[node] = -1
                    #zero_samples[self.data_test[:, node].nonzero()[0]] = -1
                    zero_samples = zero_samples[zero_samples >0]
                    zero_samples = np.random.choice(zero_samples,
                                                    #int(np.ceil(len(zero_samples)/(10*len(in_e)))),
                                                    int(np.ceil(len(zero_samples)/mask[node, 1])),
                                                    replace=False)
                    # Weaker results !
                    #step = int(np.ceil(len(zero_samples)/mask[node, 1]))
                    #if len(mask_pos[node][1]) == 0:
                    #    zero_samples = np.random.choice(zero_samples,
                    #                                    int(np.ceil(len(zero_samples)/mask[node, 1])),
                    #                                    replace=False)
                    #else:
                    #    _p = np.random.choice(mask_pos[node][1],1).item()
                    #    mask_pos[node][1].remove(_p)
                    #    zero_samples = zero_samples[_p:_p+step]

                    if len(zero_samples) == 0:
                        continue

                    for target in zero_samples:
                        yield target, node, 0

            elif set_index == 1:
                # Sample from links
                if len(neigs[node][0]) == 0:
                    continue

                node_info = {'vertex':node, 'direction':0}
                yield str(set_index), node_info, symmetric_scale*N*len(neigs[node][0])

                for target in neigs[node][0]:
                    yield node, target, weights[node, target]
            elif set_index == 2:
                # Sample from links
                if len(neigs[node][1]) == 0:
                    continue

                node_info = {'vertex':node, 'direction':1}
                yield str(set_index), node_info, symmetric_scale*N*len(neigs[node][1])

                for target in neigs[node][1]:
                    yield target, node, weights[target, node]
            else:
                raise ValueError('Set index error: %s' % set_index)




