#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


USAGE = '''build_model [-vhswp] [-k [rvalue] [-n N] [-d basedir] [-lall] [-l type] [-m model] [-c corpus] [-i iterations]
'''

from pymake.zymake import Zymake

if __name__ == '__main__':

    ##### Experience Settings
    spec = dict(
        corpus = 'clique2',
        model  = 'immsb',
        hyper       = 'auto',
        K           = [5],
        N           = 10,
        iterations  = 2,
    )

    data, model = Zymake(spec)
    model.fit(data)
    model.predict()
