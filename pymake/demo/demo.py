#!/usr/bin/python -u
# -*- coding: utf-8 -*-

from pymake import Zymake

USAGE = '''build_model [-vhswp] [-k [rvalue] [-n N] [-d basedir] [-lall] [-l type] [-m model] [-c corpus] [-i iterations]
'''

    ##### Experience Settings
    Expe = dict(
        corpus_name = 'clique2',
        model_name  = 'immsb',
        hyper       = 'auto',
        K           = 3,
        N           = 10,
        chunk       = 10000,
        iterations  = 2,
        homo        = 0, # learn W in IBP
    )


   data, model =  Zymake(spec)
