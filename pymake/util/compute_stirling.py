#!/usr/bin/env python
import os
import sys

import numpy as np
import pickle
from scipy.misc import logsumexp
try:
    from sympy.functions.combinatorial.numbers import stirling
    import sympy as sym
except:
    pass

from pymake.util.utils import get_pymake_settings


def load_stirling(style='npy'):
    stirling_path = get_pymake_settings('project_stirling')
    fn = os.path.join(stirling_path,'stirling.npy')
    npy_exists = os.path.isfile(fn)
    if style == 'npy' and npy_exists:
        return np.load(fn)
    else:
        stirlg = lookup_stirling()
        return stirlg.load()

class lookup_stirling(object):
    def __init__(self, k=1000, fn='stirling.pk' ):
        sys.setrecursionlimit(100000)
        self.fn = os.path.join(os.path.dirname(__file__),fn)
        self.k_max = k
        self.kind = 1

    def _reset(self, k):
        self.k_max = k
        self.array_stir = np.ones((k, k)) * np.inf
        self.array_stir[0,0] = 0

    def _stirling_table_dishe(self, n, m):
        if m > n:
            return np.inf
        else:
            return sym.log(stirling(n, m, kind=self.kind)).evalf()

    def run(self, k=None, save=True):
        _f = open(self.fn, 'wb')
        k_max = k or self.k_max
        self._reset(k_max)

        # Run stirling computation for (n,m) matrix equal to stirling(n,m) if n <= m else 0
        for n in range(k_max):
            print( n)
            self.array_stir[n, 1:] = np.array([ self._stirling_table_dishe(n, m) for m in range(1, k_max) ])
            if save:
                pickle.dump(self.array_stir[n], _f, protocol=pickle.HIGHEST_PROTOCOL)
                _f.flush()

        _f.close()
        return self.array_stir

    def load(self, fn=None):
        fn = fn or self.fn
        array_stir = np.array([])
        with open(fn, "rb") as f:
            array_stir = np.hstack((array_stir, pickle.load(f)))
            while True:
                try:
                    # Don't get why call to load is needed twice, but get 0.0 otherwise
                    array_stir = np.vstack((array_stir, pickle.load(f)))
                except EOFError: break
                except ValueError: break

        self.k_max = len(array_stir)
        self.array_stir = array_stir
        return self.array_stir

    def recursive_line(self, new_line=5246):
        stir = self.load()
        J = stir.shape[0]
        K = stir.shape[1]
        for x in range(new_line):
            n = J + x
            new_l =  np.ones((1, K)) * np.inf
            print(n)
            for m in range(1,K):
                if m > n:
                    continue
                elif m == n:
                    new_l[0, m] = 0
                elif m == 1:
                    new_l[0, 1] = logsumexp( [  np.log(n-1) + stir[n-1, m] ] )
                else:
                    new_l[0, m] = logsumexp( [ stir[n-1, m-1] , np.log(n-1) + stir[n-1, m] ] )
            stir = np.vstack((stir, new_l))

        #np.save('stirling.npy', stir)
        #np.load('stirling.npy')
        return stir

    def recursive_row(self, new_row=''):
        stir = np.load('stirling.npy')
        J = stir.shape[0]
        K = stir.shape[1]
        x = 0
        while stir.shape[0] != stir.shape[1]:
            m = K + x
            new_c =  np.ones((J, 1)) * np.inf
            stir = np.hstack((stir, new_c))
            print(m)
            for n in range(K,J):
                if m > n:
                    continue
                elif m == n:
                    stir[n, m] = 0

                else:
                    stir[n,m] = logsumexp( [ stir[n-1, m-1] , np.log(n-1) + stir[n-1, m] ] )
            x += 1

        #np.save('stirling.npy', stir)
        #np.load('stirling.npy',)
        return stir


if __name__ == '__main__':
    stirlg = lookup_stirling(fn = 'stirling.pk')

    # Test
    ##a = stirlg.run(k = 1000)
    #b = stirlg.load()
    b = stirlg.recursive_row()
    #try:
    #    test1 = np.array_equal(a,b)
    #    print 'Test1 succesfully passed: %s' % (test1)
    #except:
    #    pass

    #for n in range(stirlg.k_max):
    #    test2 = True
    #    for m in range(stirlg.k_max):
    #        t = sym.log(stirling(n, m, kind=1)).evalf()
    #        if t == sym.zoo:
    #            t = np.inf

    #        if t != b[n,m] :
    #            print n,m, t, b[n, m]
    #            test2 = False
    #try:
    #    print 'Test2 succesfully passed: %s' % (test2)
    #except:
    #    pass
    print(b.shape)
