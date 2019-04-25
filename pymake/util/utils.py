import sys, os
from datetime import datetime
from collections import defaultdict
import logging
import hashlib
import json
from string import Template

import numpy as np
import scipy as sp

from builtins import input
basestring = (str, bytes)


try:
    from terminal import colorize
    colored = lambda *x : str(colorize(x[0], x[1]))
except ImportError:
    lgg = logging.getLogger('root')
    lgg.debug("needs `terminal' module for colors printing")
    colored = lambda *x : x[0]


class PmkTemplate(Template):
    delimiter = '$$'
    #idpattern = r'[a-z][_a-z0-9]*'


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

def get_dest_opt_filled(parser):
    ''' Return the {dest} name of the options filled in the command line

        Parameters
        ----------
        parser : ArgParser

        Returns
        -------
        set of string
    '''

    opts_in = [opt for opt in sys.argv if opt.startswith('-') and opt not in ['-vv','-vvv']]
    opt2dest_dict = dict( (opt, act.dest) for act in parser._get_optional_actions() for opt in act.option_strings )
    dests_in = set([opt2dest_dict[opt] for opt in opts_in])
    return dests_in

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


def drop_zeros(a_list):
    #return [i for i in a_list if i>0]
    return filter(lambda x: x != 0, a_list)

def nxG(y):
    import networkx as nx
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

#
#
# Common/Utils
#
#

def retrieve_git_info():
    git_branch = subprocess.check_output(['git', 'rev-parse','--abbrev-ref' ,'HEAD']).strip().decode()
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()

    return {'git_branch':git_branch, 'git_hash':git_hash}

def hash_objects(obj, algo='md5'):
    """ Return a list of hash of the input object """
    hashalgo = getattr(hashlib, algo)

    """ Return a hash of the input """
    if isinstance(obj, (np.ndarray, list, tuple)):
        # array of int
        hashed_obj = hashalgo(np.asarray(obj).tobytes()).hexdigest()
    elif isinstance(obj, str):
        hashed_obj = hashalgo(obj.encode("utf-8")).hexdigest()
    elif isinstance(obj, dict):
        hashed_obj = hashalgo(json.dumps(obj, sort_keys=True).encode('utf8')).hexdigest()
    else:
        raise TypeError('Type of object unashable: %s' % (type(obj)))

    return hashed_obj

def ask_sure_exit(question):

    while True:
        a = input(question+' ').lower()
        if a in ('yes', 'y'):
            break
        elif a in ('no', 'n'):
            exit(2)
        else:
            print("Enter either [y|n]")

def make_path(f):
    bdir = os.path.dirname(f)
    if not os.path.exists(bdir) and bdir:
        os.makedirs(bdir)
    #fn = os.path.basename(bdir)
    #if not os.path.exists(fn) and fn:
    #    open(fn, 'a').close()
    return bdir



def Now():
    return  datetime.now()
def nowDiff(last):
    return datetime.now() - last
def ellapsed_time(text, since):
    current = datetime.now()
    delta = current - since
    print(text + ' : %s' % (delta))
    return current

def tail(filename, n_lines):
    _tail = []
    for i, line in enumerate(reverse_readline(filename)):
        if i == n_lines:
            break
        _tail.append(line)
    return _tail[::-1]
#import mmap
#def tail(filename, nlines):
#    """Returns last n lines from the filename. No exception handling"""
#    size = os.path.getsize(filename)
#    with open(filename, "rb") as f:
#        # for Windows the mmap parameters are different
#        fm = mmap.mmap(f.fileno(), 0, mmap.MAP_SHARED, mmap.PROT_READ)
#        try:
#            for i in range(size - 1, -1, -1):
#                if fm[i] == '\n':
#                    nlines -= 1
#                    if nlines == -1:
#                        break
#            return fm[i + 1 if i else 0:].splitlines()
#        finally:
#            pass
#

def reverse_readline(filename, buf_size=8192):
    """a generator that returns the lines of a file in reverse order"""
    with open(filename) as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        file_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size))
            remaining_size -= buf_size
            lines = buffer.split('\n')
            # the first line of the buffer is probably not a complete line so
            # we'll save it and append it to the last line of the next buffer
            # we read
            if segment is not None:
                # if the previous chunk starts right from the beginning of line
                # do not concact the segment to the last line of new chunk
                # instead, yield the segment first
                if buffer[-1] is not '\n':
                    lines[-1] += segment
                else:
                    yield segment
            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if len(lines[index]):
                    yield lines[index]
        # Don't yield None if the file was empty
        if segment is not None:
            yield segment


class defaultdict2(defaultdict):
	def __missing__(self, key):
		if self.default_factory is None:
			raise KeyError( key )
		else:
			ret = self[key] = self.default_factory(key)
			return ret


