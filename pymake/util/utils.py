# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os
from os.path import dirname
from datetime import datetime
from collections import defaultdict
import logging
import hashlib

import numpy as np
import scipy as sp

# __future__
try: from builtins import input
except ImportError: input = raw_input
try: basestring = basestring # python2
except NameError: basestring = (str, bytes) # python3


try:
    from terminal import colorize
    colored = lambda *x : str(colorize(x[0], x[1]))
except ImportError:
    lgg = logging.getLogger('root')
    lgg.debug("needs `terminal' module for colors printing")
    colored = lambda *x : x[0]

#from itertools import cycle
class Cycle(object):
    def __init__(self, seq):
        self.seq = seq
        self.it = np.nditer([seq])
    def next(self):
        return self.__next__()
    def __next__(self):
        try:
            return next(self.it).item()
        except StopIteration:
            self.it.reset()
            # Exception on nditer when seq is empty (infinite recursivity)
            return self.next()

    def reset(self):
        return self.it.reset()

    def copy(self):
        return self.__class__(self.seq)

def ask_sure_exit(question):

    while True:
        a = input(question+' ').lower()
        if a == 'yes':
            break
        elif a == 'no':
            exit()
        else:
            print("Enter either yes/no")

def get_dest_opt_filled(parser):
    ''' Return the {dest} name of the options filled in the command line

        Parameters
        ----------
        parser : ArgParser

        Returns
        -------
        set of string
    '''

    opts_in = [opt for opt in sys.argv if opt.startswith('-')]
    opt2dest_dict = dict( (opt, act.dest) for act in parser._get_optional_actions() for opt in act.option_strings )
    dests_in = set([opt2dest_dict[opt] for opt in opts_in])
    return dests_in

def Now():
    return  datetime.now()
def nowDiff(last):
    return datetime.now() - last
def ellapsed_time(text, since):
    current = datetime.now()
    delta = current - since
    print(text + ' : %s' % (delta))
    return current

def jsondict(d):
    if isinstance(d, dict):
        return {str(k):v for k,v in d.items()}
    return d

def parse_file_conf(fn, sep=':', comments=('#','%')):
    with open(fn) as f:
        parameters = f.read()
    parameters = filter(None, parameters.split('\n'))
    parameters = dict((p[0].strip(), p[1].strip()) for p in (t.strip().split(sep) for t in parameters if not t.strip().startswith(comments)))
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

def make_path(f):
    bdir = os.path.dirname(f)
    if not os.path.exists(bdir) and bdir:
        os.makedirs(bdir)
    #fn = os.path.basename(bdir)
    #if not os.path.exists(fn) and fn:
    #    open(fn, 'a').close()
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
        G = nx.from_numpy_matrix(y, create_using=typeG)
    else:
        G = y
    return G


# Global settings
__default_config = defaultdict(lambda: '', dict(project_data = os.path.expanduser('~/.pymake/data'),
                                                  project_figs = os.path.expanduser('~/.pymake/results/figs') ,
                                                  default_spec = 'pymake.spec',
                                                  default_script = 'pymake.script',
                                                  default_model = 'pymake.model',
                                                  default_corpus = '?')
                               )
def set_global_settings(settings, default_config=__default_config, cfg_name='pymake.cfg'):
    _settings = default_config.copy()
    _settings.update(settings)
    ctnt = '\n'.join(('{0} = {1}'.format(k,v) for k,v in _settings.items()))
    cfg_file = os.path.join(os.getenv('PWD') , cfg_name)
    with open(cfg_file, 'wb') as _f:
        return _f.write(ctnt.encode('utf8'))

def get_global_settings(key=None, default_config=__default_config, cfg_name='pymake.cfg'):
    #dir =  os.path.dirname(os.path.realpath(__file__))
    #dir = os.getcwd()
    dir = os.getenv('PWD')
    cfg_file = os.path.join(dir, cfg_name)

    if not os.path.isfile(cfg_file):
        cfg_file = os.path.join(os.path.expanduser('~') ,'.pymake', cfg_name)
        if not os.path.isfile(cfg_file):
            dir_cfg = make_path(cfg_file)
            ctnt = '\n'.join(('{0} = {1}'.format(k,v) for k,v in  default_config.items()))
            with open(cfg_file, 'wb') as _f:
                _f.write(ctnt.encode('utf8'))

    config = parse_file_conf(cfg_file, sep='=')

    if not key:
        settings =  config
    elif key.startswith('_'):
        res = []
        for k in ['default'+key, 'contrib'+key]:
            res += os.path.expanduser(config.get(k, default_config[k])).split(',')
        settings =  [e for e in map(str.strip, res) if e]
    else:
        settings = os.path.expanduser(config.get(key, default_config[key]))

    #print(settings)
    return settings

def retrieve_git_info():
    git_branch = subprocess.check_output(['git', 'rev-parse','--abbrev-ref' ,'HEAD']).strip().decode()
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()

    return {'git_branch':git_branch, 'git_hash':git_hash}

def hash_object(obj, algo='md5'):
    """ Return a list of hash of the input object """
    hashalgo = getattr(hashlib, algo)

    """ Return a hash of the input """
    if isinstance(obj, (np.ndarray, list, tuple)):
        # array of int
        hashed_obj = hashalgo(np.asarray(obj).tobytes()).hexdigest()
    elif type(obj) is str:
        hashed_obj = hashalgo(obj.encode("utf-8")).hexdigest()
    else:
        raise TypeError('Type of object unashable: %s' % (type(obj)))

    return hashed_obj



