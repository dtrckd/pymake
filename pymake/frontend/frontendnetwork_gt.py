import os

from .frontend import DataBase
from .drivers import OnlineDatasetDriver

from pymake.util.math import *

try:
    import graph_tool as gt
    from graph_tool import collection as net_collection
    from graph_tool import clustering, inference, spectral
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
    def _get_data(cls, expe, corpus=None):
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
        weighted = corpus.get('weighted', False)
        N = corpus.get('nodes')
        E = corpus.get('edges')

        g = gt.Graph(directed=directed)
        #g.add_vertex(N)

        labels = g.new_vertex_property("int") # g.new_vp
        weights = g.new_edge_property("int") # g.new_ep

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

        g.vertex_properties['labels'] = labels # g.vp
        g.edge_properties['weights'] = weights # g.ep

        # Remove first index in case of indexation start at 1
        zero_degree = g.vertex(0).out_degree() + g.vertex(0).in_degree()
        if zero_degree == 0:
            g.remove_vertex(0)

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
    def from_expe(cls, expe, load=True, corpus=None, save=True):
        if '_force_load_data' in expe:
            load = expe._force_load_data
        if '_force_save_data' in expe:
            save = expe._force_save_data

        fn = cls._resolve_filename(expe)
        target_file_exists = os.path.exists(fn)

        if expe.corpus in net_collection.data:
            data = gt.collection.data[expe.corpus]
        elif load is False or not target_file_exists:
            data = cls._get_data(expe, corpus=corpus)
            if save:
                # ===== save ====
                cls._save_data(fn, data)
        else:
            # ===== load ====
            data = cls._get_data(expe, corpus=corpus)
            data = cls._load_data(fn)

        frontend = cls(expe)
        frontend.data = data

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
        return gt.topology.pseudo_diameter(self.data)

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
        return '%s / min-max value: %s, %s' % (self._net_type,
                                               weights.min(),
                                               weights.max())

    def feat_len(self):
        weights = self.data.ep['weights'].a
        return len(np.unique(weights))


