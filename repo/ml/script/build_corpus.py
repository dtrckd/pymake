from pymake.frontend.frontendtext import frontendText
# @Issue43: Parser/config unification.
from pymake.util.utils import Now, ellapsed_time, ask_sure_exit
from pymake.util.argparser import argparser
import os

import numpy as np
import scipy as sp
np.set_printoptions(threshold='nan')

''' Build Text Corpus '''

corpuses = ('nips12',)
corpuses = ('nips12', 'kos','reuter50', 'nips', 'enron', 'nytimes', 'pubmed', '20ngroups')

if __name__ == '__main__':
    config = dict(
        _load_data = False,
        _save_data = True,
    )
    config.update(argparser.gramexp())

    ############################################################
    ##### Simulation Output
    if config.get('simulate'):
        print ('''--- Simulation settings ---
        Build Corpuses %s''' % (str(corpuses)))
        exit()

    ask_sure_exit('Sure to overwrite Corpus / Text ?')

    fn_corpus_build = os.path.join(config['bdir'], 'text','Corpuses.txt')
    _f = open(fn_corpus_build, 'a')
    _f.write('/**** %s ****/\n\n' % (Now()))

    for corpus_name in corpuses:
        startt = Now()
        frontend = frontendText(config)
        frontend.load_data(corpus_name)
        building_corpus_time = (ellapsed_time('Prepropressing %s'%corpus_name, startt) - startt)
        prop = frontend.get_data_prop()
        prop.update(time='%0.3f' % (building_corpus_time.total_seconds()/60) )
        msg = frontend.template(prop)
        print (msg)
        _f.write(msg)
        _f.flush()


