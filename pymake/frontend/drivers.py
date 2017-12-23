from numpy import ma
import numpy as np

import pandas as pd

class DatasetDriver(object):
    ''' Parse dataset file using pandas'''

    _comment = '%'

    # No pandas here....
    def parse_tnet(self, fn, sep=' '):
        ''' Grammar retro-ingennired from fb/emaileu.txt '''
        self.log.debug('opening file: %s' % fn)
        with open(fn) as f:
            content = f.read()
        lines = list(filter(None, content.split('\n')))
        line1_length = lines[0].strip().split(sep)
        edges = {}
        if len(line1_length) == 2:
            # format 'i j' if edges.
            self._data_file_format = 'txt'
            for line in lines:
                dyad = line.strip().split(sep)[:]
                dyad = '.'.join(dyad)
                edges[dyad] = edges.get(dyad, 0) + 1
            #edges = [l.strip().split(sep)[:] for l in lines]
        elif len(line1_length) == 5:
            # format '"date" i j weight'.
            self._data_file_format = 'tnet'
            for line in lines:
                _line = line.strip().split(sep)
                dyad = _line[-3:-1]
                dyad = '.'.join(dyad)
                w = int(_line[-1])
                edges[dyad] = edges.get(dyad, 0) + w
            #edges = [l.strip().split(sep)[-3:-1] for l in lines]

        edges = np.array([ (e.split('.')[0], e.split('.')[1], w+1) for e, w in edges.items()], dtype=int) -1

        edges[:, 0:2] -= edges[:, 0:2].min()
        N = edges[:, 0:2].max()+1

        g = np.zeros((N,N))
        g[tuple(edges[:, :2].T)] = edges[:, 2]
        return g

    # No pandas here....
    def parse_csv(self, fn, sep=';'):
        ''' Grammar retro-ingennired from manufacturing.csv '''
        self.log.debug('opening file: %s' % fn)
        with open(fn, 'r') as f:
            content = f.read()
        lines = list(filter(None, content.split('\n')))[1:]
        edges = {}
        for line in lines:
            dyad = line.strip().split(sep)[0:2]
            dyad = '.'.join(dyad)
            edges[dyad] = edges.get(dyad, 0) + 1
        #edges = [l.strip().split(sep)[0:2] for l in lines]
        #edges = np.array([ (e[0], e[1]) for e in edges], dtype=int) -1
        edges = np.array([ (e.split('.')[0], e.split('.')[1], w+1) for e, w in edges.items()], dtype=int) -1

        edges[:, 0:2] -= edges[:, 0:2].min()
        N = edges[:, 0:2].max()+1

        g = np.zeros((N,N))
        g[tuple(edges[:, :2].T)] = edges[:, 2]
        return g


    def parse_dancer(self, fn, sep=';'):
        """ Parse Network data depending on type/extension """
        self.log.debug('opening file: %s' % fn)

        data = pd.read_csv(fn, sep=sep, names=['n', 'feat', 'cluster' ], comment=self._comment)
        parameters = data.dropna()
        self.clusters = parameters['cluster'].values
        self.features = np.array([list(map(float, f.split('|'))) for f in parameters['feat'].values])

        data = data.ix[data['cluster'].isna()]
        data['cluster'] = 1 # <= the weight
        data = data.loc[pd.to_numeric(data['n'], errors='coerce').dropna().index].as_matrix().astype(int)

        data[:, 0:2] -= data[:, 0:2].min()
        N = data[:, 0:2].max()+1
        y = np.zeros((N,N))
        e_l = data[:,2] > 0
        e_ix = data[:, 0:2][e_l]
        ix = list(zip(*e_ix))
        y[ix] = data[:,2][e_l]
        return y

    def parse_dat(self, fn, sep="\s+"):
        """ Parse Network data depending on type/extension """
        self.log.debug('opening file: %s' % fn)

        def _row_len(fn):
            ''' Seek for the length of the csv row, then break quicly '''
            f = open(fn, 'rb')
            inside = {'vertices':False, 'edges':False }
            data  = []
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
                        continue
                elif line.startswith(('DATA','*edges' )) or inside['edges']:
                    if not inside['edges']:
                        inside['edges'] = True # break
                        continue
                    if line.startswith('#') or not line.strip() or len(line.split()) < 2 :
                        inside['edges'] = False
                    else:
                        # Parsing assignation
                        data.append( line.split() )
                        break
            f.close()
            return len(data[0])

        # Sender, Reiceiver, Edges
        row_len = _row_len(fn)
        if  row_len == 3:
            cols = ['s', 'r', 'weight']
        elif row_len == 2:
            cols = ['s', 'r']
        else:
            raise ValueError('I/O error for dataset file: %s' % fn)

        data = pd.read_csv(fn, sep=sep, names=cols, comment=self._comment)
        if len(cols) == 2:
            data['weight'] = np.ones(data.shape[0])
            cols = ['s', 'r', 'weight']

        cond = pd.to_numeric(data['s'], errors='coerce').dropna().index & pd.to_numeric(data['r'], errors='coerce').dropna().index
        data = data.loc[cond].as_matrix().astype(int)


        data[:, 0:2] -= data[:, 0:2].min()
        N = data[:, 0:2].max()+1
        y = np.zeros((N,N))
        e_l = data[:,2] > 0
        e_ix = data[:, 0:2][e_l]
        ix = list(zip(*e_ix))
        y[ix] = data[:,2][e_l]
        return y



class RawDatasetDriver(object):
    ''' Parse dataset file using python loop (deprecated) '''


    def parse_tnet(self, fn, sep=' '):
        ''' Grammar retro-ingennired from fb/emaileu.txt '''
        self.log.debug('opening file: %s' % fn)
        with open(fn) as f:
            content = f.read()
        lines = list(filter(None, content.split('\n')))
        line1_length = lines[0].strip().split(sep)
        edges = {}
        if len(line1_length) == 2:
            # format 'i j' if edges.
            self._data_file_format = 'txt'
            for line in lines:
                dyad = line.strip().split(sep)[:]
                dyad = '.'.join(dyad)
                edges[dyad] = edges.get(dyad, 0) + 1
            #edges = [l.strip().split(sep)[:] for l in lines]
        elif len(line1_length) == 5:
            # format '"date" i j weight'.
            self._data_file_format = 'tnet'
            for line in lines:
                _line = line.strip().split(sep)
                dyad = _line[-3:-1]
                dyad = '.'.join(dyad)
                w = int(_line[-1])
                edges[dyad] = edges.get(dyad, 0) + w
            #edges = [l.strip().split(sep)[-3:-1] for l in lines]

        edges = np.array([ (e.split('.')[0], e.split('.')[1], w+1) for e, w in edges.items()], dtype=int) -1
        N = edges.max() +1
        #N = max(list(itertools.chain(*edges))) + 1

        g = np.zeros((N,N))
        g[tuple(edges[:, :2].T)] = edges[:, 2]
        return g

    def parse_csv(self, fn, sep=';'):
        ''' Grammar retro-ingennired from manufacturing.csv '''
        self.log.debug('opening file: %s' % fn)
        with open(fn, 'r') as f:
            content = f.read()
        lines = list(filter(None, content.split('\n')))[1:]
        edges = {}
        for line in lines:
            dyad = line.strip().split(sep)[0:2]
            dyad = '.'.join(dyad)
            edges[dyad] = edges.get(dyad, 0) + 1
        #edges = [l.strip().split(sep)[0:2] for l in lines]
        #edges = np.array([ (e[0], e[1]) for e in edges], dtype=int) -1
        edges = np.array([ (e.split('.')[0], e.split('.')[1], w+1) for e, w in edges.items()], dtype=int) -1
        N = edges.max() +1
        #N = max(list(itertools.chain(*edges))) + 1

        g = np.zeros((N,N))
        g[tuple(edges[:, :2].T)] = edges[:, 2]
        return g

    def parse_dancer(self, fn, sep=';'):
        """ Parse Network data depending on type/extension """
        self.log.debug('opening file: %s' % fn)
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
        g = np.zeros((N,N))
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
        self.log.debug('opening file: %s' % fn)
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
        edges = {}
        if row_size == 2:
            # like .txt
            for line in data:
                dyad = line.strip().split(sep)[:]
                dyad = '.'.join(dyad)
                edges[dyad] = edges.get(dyad, 0) + 1
        elif row_size == 3:
            for line in data:
                _line = line.strip().split(sep)
                dyad = _line[0:2]
                dyad = line.strip().split(sep)[:]
                dyad = '.'.join(dyad)
                w = int(_line[-1]) # can be zeros
                edges[dyad] = edges.get(dyad, 0) + int(w)
        else:
            raise NotImplementedError

        edges = np.array([ (e.split('.')[0], e.split('.')[1], w+1) for e, w in edges.items()], dtype=int) -1
        N = edges.max() +1
        g = np.zeros((N,N))
        g[tuple(edges[:, :2].T)] = edges[:, 2]
        return g

