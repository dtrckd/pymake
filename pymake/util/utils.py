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

def ask_sure_exit(question):

    while True:
        a = input(question+' ').lower()
        if a == 'yes':
            break
        elif a == 'no':
            exit()
        else:
            print("Enter either yes/no")

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

def parse_file_conf(fn, sep=':'):
    with open(fn) as f:
        parameters = f.read()
    parameters = filter(None, parameters.split('\n'))
    parameters = dict((p[0].strip(), p[1].strip()) for p in (t.strip().split(sep) for t in parameters))
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
        G = nx.from_numpy_matrix(y, create_using=typeG)
    else:
        G = y
    return G
