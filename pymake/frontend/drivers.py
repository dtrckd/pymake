from numpy import ma
import numpy as np
import logging

try:
    import pandas as pd
except Exception as e:
    print('Error while importing pandas: %s' % e)



class DatasetDriver(object):

    ''' Parse dataset file using pandas'''

    _comment = '%'
    log = logging.getLogger('root')

    # No pandas here....
    @classmethod
    def parse_tnet(cls, fn, sep=' '):
        ''' Grammar retro-ingennired from fb/emaileu.txt. tnet format is official ? '''
        cls.log.debug('opening file: %s' % fn)
        with open(fn) as f:
            content = f.read()
        lines = list(filter(None, content.split('\n')))
        line1_length = lines[0].strip().split(sep)
        edges = {}
        if len(line1_length) == 2:
            # format 'i j' if edges.
            data_file_format = 'txt'
            for line in lines:
                dyad = line.strip().split(sep)
                dyad = '.'.join(dyad)
                edges[dyad] = edges.get(dyad, 0) + 1
            #edges = [l.strip().split(sep) for l in lines]
        elif len(line1_length) == 5:
            # format '"date" i j weight'.
            data_file_format = 'tnet'
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
        data = dict(data=g)
        return data

    # No pandas here....
    @classmethod
    def parse_csv(cls, fn, sep=';'):
        ''' Grammar retro-ingennired from manufacturing.csv '''
        cls.log.debug('opening file: %s' % fn)
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
        data = dict(data=g)
        return data


    @classmethod
    def parse_dancer(cls, fn, sep=';'):
        """ Parse Network data depending on type/extension """
        cls.log.debug('opening file: %s' % fn)

        data = pd.read_csv(fn, sep=sep, names=['n', 'feat', 'cluster' ], comment=cls._comment)
        parameters = data.dropna()
        clusters = parameters['cluster'].values.astype(int)
        features = np.array([list(map(float, f.split('|'))) for f in parameters['feat'].values])

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

        data = dict(data=y, clusters=clusters, features=features)
        return data

    @classmethod
    def parse_dat(cls, fn, sep="\s+"):
        """ Parse Network data depending on type/extension """
        cls.log.debug('opening file: %s' % fn)

        def _row_len(fn):
            ''' Seek for the length of the csv row, then break quicly '''
            inside = {'vertices':False, 'edges':False }
            data  = []
            for _line in open(fn):
                line = _line.strip()
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
            return len(data[0])

        # Sender, Reiceiver, Edges
        row_len = _row_len(fn)
        if  row_len == 3:
            cols = ['s', 'r', 'weight']
        elif row_len == 2:
            cols = ['s', 'r']
        else:
            raise ValueError('I/O error for dataset file: %s' % fn)

        data = pd.read_csv(fn, sep=sep, names=cols, comment=cls._comment)
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
        data = dict(data=y)
        return data



class OnlineDatasetDriver(object):

    ''' Parse dataset file using pandas'''

    _comment = '%'
    log = logging.getLogger('root')

    @classmethod
    def parse_tnet(cls, fn, sep=' '):
        ''' Grammar retro-ingennired from fb/emaileu.txt. tnet format is official ? '''

        cls.log.debug('opening file: %s' % fn)

        for line in open(fn):
            line = line.strip()
            if not line:
                continue

            line1_length = line.split(sep)

            if len(line1_length) == 2:
                # format 'i j' if edges.
                data_file_format = 'txt'
                v1, v2 = line.strip().split(sep)
                w = 1
                yield int(v1), int(v2), w, None
            elif len(line1_length) == 5:
                # format '"date" i j weight'.
                data_file_format = 'tnet'
                _line = line.strip().split(sep)
                v1, v2 = _line[-3:-1]
                w = int(_line[-1])
                if w == 0:
                    continue
                else:
                    yield int(v1), int(v2), w, None


    @classmethod
    def parse_csv(cls, fn, sep=';'):
        ''' Grammar retro-ingennired from manufacturing.csv '''

        cls.log.debug('opening file: %s' % fn)

        cpt = 0
        for line in open(fn):
            if cpt == 0:
                # Ignore first status line
                cpt += 1
                continue
            v1, v2 = line.strip().split(sep)[0:2]
            w = 1
            yield int(v1), int(v2), w, None


    @classmethod
    def parse_dancer(cls, fn, sep=';'):

        cls.log.debug('opening file: %s' % fn)

        inside = {'vertices':False, 'edges':False }
        for line in open(fn):
            line = line.strip()
            if line.startswith('# Vertices') or inside['vertices']:
                if not inside['vertices']:
                    inside['vertices'] = True
                    continue
                if line.startswith('#') or not line.strip() :
                    inside['vertices'] = False # break
                else:
                    # Parsing assignation
                    elements = line.strip().split(sep)
                    index = int(elements[0])
                    clust = int(elements[-1])
                    feats = list(map(float, elements[-2].split('|')))
                    obj = {'cluster': clust, 'features': feats, 'index':index}
                    yield obj
            elif line.startswith('# Edges') or inside['edges']:
                if not inside['edges']:
                    inside['edges'] = True
                    continue
                if line.startswith('#') or not line.strip() :
                    inside['edges'] = False # break
                else:
                    # Parsing assignation
                    v1, v2 = line.split(sep)
                    w = 1
                    yield int(v1), int(v2), w, None


    @classmethod
    def parse_dat(cls, fn, sep=" "):
        """ Parse Network data depending on type/extension """

        cls.log.debug('opening file: %s' % fn)

        inside = {'vertices':False, 'edges':False }
        for line in open(fn):
            line = line.strip()
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
                    splitline = line.split(sep)
                    row_size = len(splitline)
                    if row_size == 2:
                        # like .txt
                        v1, v2 = splitline
                        w = 1
                        yield int(v1), int(v2), w, None
                    elif row_size == 3:
                        v1, v2 = splitline[0:2]
                        w = int(splitline[2])
                        if w == 0:
                            continue
                        else:
                            yield int(v1), int(v2), w, None
                    else:
                        raise NotImplementedError








class RawDatasetDriver(object):

    ''' Parse dataset file using python loop (deprecated) '''

    _comment = '%'
    log = logging.getLogger('root')

    @classmethod
    def parse_tnet(cls, fn, sep=' '):
        ''' Grammar retro-ingennired from fb/emaileu.txt '''
        cls.log.debug('opening file: %s' % fn)
        with open(fn) as f:
            content = f.read()
        lines = list(filter(None, content.split('\n')))
        line1_length = lines[0].strip().split(sep)
        edges = {}
        if len(line1_length) == 2:
            # format 'i j' if edges.
            data_file_format = 'txt'
            for line in lines:
                dyad = line.strip().split(sep)
                dyad = '.'.join(dyad)
                edges[dyad] = edges.get(dyad, 0) + 1
            #edges = [l.strip().split(sep) for l in lines]
        elif len(line1_length) == 5:
            # format '"date" i j weight'.
            data_file_format = 'tnet'
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
        data = dict(data=g)
        return data

    @classmethod
    def parse_csv(cls, fn, sep=';'):
        ''' Grammar retro-ingennired from manufacturing.csv '''
        cls.log.debug('opening file: %s' % fn)
        with open(fn) as f:
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
        data = dict(data=g)
        return data

    @classmethod
    def parse_dancer(cls, fn, sep=';'):
        """ Parse Network data depending on type/extension """
        cls.log.debug('opening file: %s' % fn)
        data = []
        inside = {'vertices':False, 'edges':False }
        clusters = []
        features = []
        for line in open(fn):
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
                    inside['edges'] = False # break
                else:
                    # Parsing assignation
                    data.append( line.strip() )

        edges = np.array([tuple(row.split(sep)) for row in data]).astype(int)
        g = np.zeros((N,N))
        g[[e[0] for e in edges], [e[1] for e in edges]] = 1
        g[[e[1] for e in edges], [e[0] for e in edges]] = 1
        # ?! .T

        try:
            parameters = parse_file_conf(os.path.join(os.path.dirname(fn), 'parameters'))
            parameters['devs'] = list(map(float, parameters['devs'].split(sep)))
        except IOError:
            parameters = {}
        finally:
            # @Obsolete !
            parameters_ = parameters

        clusters = clusters
        features = np.array(features)
        data = dict(data=g, clusters=clusters, features=features)
        return data

    @classmethod
    def parse_dat(cls, fn, sep=' '):
        """ Parse Network data depending on type/extension """
        cls.log.debug('opening file: %s' % fn)
        data = []
        inside = {'vertices':False, 'edges':False }
        for _line in open(fn):
            line = _line.strip()
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

        row_size = len(data[0].split(sep))
        edges = np.array([tuple(row.split(sep)) for row in data]).astype(int)-1
        edges = {}
        if row_size == 2:
            # like .txt
            for line in data:
                dyad = line.strip().split(sep)
                dyad = '.'.join(dyad)
                edges[dyad] = edges.get(dyad, 0) + 1
        elif row_size == 3:
            for line in data:
                _line = line.strip().split(sep)
                dyad = _line[0:2]
                dyad = '.'.join(dyad)
                w = int(_line[-1]) # can be zeros
                edges[dyad] = edges.get(dyad, 0) + int(w)
        else:
            raise NotImplementedError

        edges = np.array([ (e.split('.')[0], e.split('.')[1], w+1) for e, w in edges.items()], dtype=int) -1
        N = edges.max() +1
        g = np.zeros((N,N))
        g[tuple(edges[:, :2].T)] = edges[:, 2]
        data = dict(data=g)
        return data

