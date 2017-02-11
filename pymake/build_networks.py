#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

# @Issue43: Parser/config unification.
from util.utils import Now, ellapsed_time, ask_sure_exit
from util.argparser import argparser
import os

import numpy as np
import scipy as sp
np.set_printoptions(threshold='nan')

from frontend.frontendnetwork import frontendNetwork
from expe.spec import _spec


''' Build Networks Corpus '''

# Expe
corpuses = ('generator7',)
corpuses = ( 'generator7', 'generator12', 'generator10', 'generator4')
corpuses += ( 'fb_uc', 'manufacturing', )

corpuses = _spec.CORPUS_NET_ALL

if __name__ == '__main__':
    config = dict(
        ##### Global settings
        ###### I/O settings
        bdir = '../data',
        load_data = False,
        save_data = True,
    )
    config.update(argparser.gramexp())

    ############################################################
    ##### Simulation Output
    if config.get('simulate'):
        print ('''--- Simulation settings ---
        Build Corpuses %s''' % (str(corpuses)))
        exit()

    ask_sure_exit('Sure to overwrite Graph / networks ?')

    fn_corpus_build = os.path.join(config['bdir'], 'networks', 'Corpuses.txt')
    _f = open(fn_corpus_build, 'a')
    _f.write('/**** %s ****/\n\n' % (Now()))

    for corpus_name in corpuses:
        startt = Now()
        frontend = frontendNetwork(config)
        frontend.load_data(corpus_name)
        building_corpus_time = (ellapsed_time('Prepropressing %s'%corpus_name, startt) - startt)
        prop = frontend.get_data_prop()
        prop.update(time='%0.3f' % (building_corpus_time.total_seconds()/60) )
        msg = frontend.template(prop)
        print (msg)
        _f.write(msg)
        _f.flush()

