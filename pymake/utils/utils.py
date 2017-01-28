# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
try: from builtins import input
except ImportError: input = raw_input

import sys, os
from os.path import dirname
from datetime import datetime
from collections import defaultdict
import logging

import numpy as np
import scipy as sp

#from itertools import cycle
class Cycle(object):
    def __init__(self, seq):
        self.seq = seq
        self.it = np.nditer([seq])
    def next(self):
        try:
            return self.it.next().item()
        except StopIteration:
            self.it.reset()
            # Exception on nditer when seq is empty (infinite recursivity)
            return self.next()
    def reset(self):
        return self.it.reset()

# use https://github.com/kennethreitz/args
def argParse(usage="Usage ?"):
    argdict = defaultdict(lambda: False)
    for i, arg in enumerate(sys.argv):
        if arg in ('-np', '--no-print', '--printurl'):
            argdict['noprint'] = True
        elif arg in ('-w',):
            argdict.update(write = True)
        elif arg in ('-nv',):
            argdict.update(verbose = logging.WARN)
        elif arg in ('-v',):
            argdict.update(verbose = logging.DEBUG)
        elif arg in ('-vv',):
            argdict.update(verbose = logging.CRITICAL)
        elif arg in ('-p',):
            argdict.update(predict = 1)
        elif arg in ('-s',):
            argdict['simul'] = arg
        elif arg in ('-nld',):
            argdict['load_data'] = False
        elif arg in ('--seed',):
            argdict['seed'] = 42
        elif arg in ('-n', '--limit'):
            # no int() because could be all.
            _arg = sys.argv.pop(i+1)
            argdict['N'] = _arg
        elif arg in ('--alpha', '--hyper'):
            _arg = sys.argv.pop(i+1)
            argdict['hyper'] = _arg
        elif arg in ('--hyper_prior', ):
            _arg = []
            while(sys.argv[i+1].isdigit()):
                _arg.append(int(sys.argv.pop(i+1)))
            argdict['hyper_prior'] = _arg
        elif arg in ('--refdir',):
            _arg = sys.argv.pop(i+1)
            argdict['refdir'] = _arg
        elif arg in ('--repeat',):
            repeat = int(sys.argv.pop(i+1))
            if repeat < 0:
                _arg = ''
            else:
                _arg = str(repeat)
            argdict['repeat'] = _arg
        elif arg in ('-k',):
            _arg = sys.argv.pop(i+1)
            argdict['K'] =int(_arg)
        elif arg in ('--homo',):
            _arg = int(sys.argv.pop(i+1))
            argdict['homo'] = _arg
        elif arg in ('-i',):
            _arg = int(sys.argv.pop(i+1))
            argdict['iterations'] = _arg
        elif arg in ('-c',):
            _arg = sys.argv.pop(i+1)
            argdict['corpus_name'] = _arg
        elif arg in ('-m',):
            _arg = sys.argv.pop(i+1)
            argdict['model_name'] = _arg
        elif arg in ('-d',):
            _arg = sys.argv.pop(i+1)
            argdict['bdir'] = _arg+'/'
        elif arg in ('-lall', ):
            argdict['lall'] = True
        elif arg in ('-l', '-load', '--load'):
            try:
                _arg = sys.argv[i+1]
            except:
                _arg = 'tmp'
            if not os.path.exists(_arg):
                if _arg == 'corpus' or _arg == 'model':
                    argdict['load'] = sys.argv.pop(i+1)
                else:
                    argdict['load'] = False
            else:
                _arg = sys.argv.pop(i+1)
                argdict['load'] = _arg
        elif arg in ('-r', '--random', 'random'):
            _arg = sys.argv.pop(i+1)
            argdict['random'] = _arg
        elif arg in ('-g'):
            argdict.update(random = False)
        elif arg in ('--help','-h'):
            print(usage)
            exit(0)
        else:
            if i == 0:
                argdict.setdefault('arg', False)
                #argdict.setdefault('arg', 'no args')
            else:
                argdict.update({arg:arg})

    return argdict

def ask_sure_exit(question):

    while True:
        a = input(question+' ').lower()
        if a == 'yes':
            break
        elif a == 'no':
            exit()
        else:
            print("Enter either yes/no")

def setup_logger(name, fmt, verbose, file=None):
    #formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    #logging.basicConfig(format='Gensim : %(message)s', level=logging.DEBUG)
    logging.basicConfig(format=fmt, level=verbose)

    formatter = logging.Formatter(fmt=fmt)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(verbose)
    logger.addHandler(handler)
    return logger


def Now():
    return  datetime.now()
def ellapsed_time(text, since):
    current = datetime.now()
    delta = current - since
    print(text + ' : %s' % (delta))
    return current

def jsondict(d):
    if isinstance(d, dict):
        return {str(k):v for k,v in d.items()}
    return d

def parse_file_conf(fn):
    with open(fn) as f:
        parameters = f.read()
    parameters = filter(None, parameters.split('\n'))
    parameters = dict((p[0], p[1]) for p  in (t.strip().split(':') for t in parameters))
    for k, v in parameters.items():
        if  '.' in v:
            try:
                parameters[k] = float(v)
            except:
                pass
        else:
            try:
                parameters[k] = int(v)
            except:
                pass
    return parameters


def getGraph(target='', **conf):
    basedir = conf.get('filen', dirname(__file__) + '/../../data/networks/')
    filen = basedir + target
    f = open(filen, 'r')

    data = []
    N = 0
    inside = [False, False]
    for line in f:
        if line.startswith('# Vertices') or inside[0]:
            if not inside[0]:
                inside[0] = True
                continue
            if line.startswith('#') or not line.strip() :
                inside[0] = False
            else:
                # Parsing assignation
                N += 1
        elif line.startswith('# Edges') or inside[1]:
            if not inside[1]:
                inside[1] = True
                continue
            if line.startswith('#') or not line.strip() :
                inside[1] = False
            else:
                # Parsing assignation
                data.append( line.strip() )
    f.close()
    edges = [tuple(row.split(';')) for row in data]
    g = np.zeros((N,N))
    g[[e[0] for e in edges],  [e[1] for e in edges]] = 1
    g[[e[1] for e in edges],  [e[0] for e in edges]] = 1
    return g



# Assign new values to an array according to a map list
def set_v_to(a, map):
    new_a = a.copy()
    for k, c in dict(map).iteritems():
        new_a[a==k] = c

    return new_a


# Re-order the confusion matrix in order to map the cluster (columns) to the best (classes) according to purity
# One class by topics !
# It modify confu and map in-place
# Return: list of tuple that map topic -> class
import sys
sys.setrecursionlimit(10000)
def map_class2cluster_from_confusion(confu, map=None, cpt=0, minmax='max'):
    assert(confu.shape[0] == confu.shape[1])

    if minmax == 'max':
        obj_f = np.argmax
    else:
        obj_f = np.argmin

    if len(confu) -1  == cpt:
        # Recursive stop condition
        return sorted(map)
    if map is None:
        confu = confu.copy()
        map = [ (i,i) for i in range(len(confu)) ]
        #map = np.array(map)

    #K = confu.shape[0]
    #C = confu.shape[1]
    previous_assigned = [i[1] for i in map[:cpt]]
    c_l = obj_f(np.delete(confu[cpt], previous_assigned))
    # Get the right id of the class
    for j in sorted(previous_assigned):
        # rectify c_l depending on which class where already assigning
        if c_l >= j:
            c_l += 1
        else:
            break
    m_l = confu[cpt, c_l]
    # Get the right id of the topic
    c_c = obj_f(confu[cpt:,c_l]) + cpt
    m_c = confu[c_c, c_l]
    if m_c > m_l:
        # Move the line corresponding to the max for this class to the top
        confu[[cpt, c_c], :] = confu[[c_c, cpt], :]
        map[cpt], map[c_c] = map[c_c], map[cpt] # Doesn't work if it's a numpy array
        return map_class2cluster_from_confusion(confu, map, cpt)
    else:
        # Map topic 1 to class c_l and return confu - topic 1 and class c_l
        map[cpt] = (map[cpt][0], c_l)
        cpt += 1
        return map_class2cluster_from_confusion(confu, map, cpt)

def make_path(bdir):
    fn = os.path.basename(bdir)
    _bdir = os.path.dirname(bdir)
    if not os.path.exists(_bdir) and _bdir:
        os.makedirs(_bdir)
    if not os.path.exists(fn) and fn:
        #open(fn, 'a').close()
        pass # do i need it
    return bdir


def drop_zeros(a_list):
    #return [i for i in a_list if i>0]
    return filter(lambda x: x != 0, a_list)

import networkx as nx
def nxG(y):
    if type(y) is np.ndarray:
        if (y == y.T).all():
            # Undirected Graph
            typeG = nx.Graph()
        else:
            # Directed Graph
            typeG = nx.DiGraph()
        G = nx.from_numpy_matrix(y, typeG)
    else:
        G = y

    return G
