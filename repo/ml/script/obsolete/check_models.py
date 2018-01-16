#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pymake.frontend.manager import ModelManager, FrontendManager
from frontend.frontendnetwork import frontendNetwork
from frontend.frontend_io import *
from expe.spec import _spec_; _spec = _spec_()
from util.argparser import argparser

""" Parse models on disk, for checking
    or updating results
"""

USAGE = '''\
# Usage:
    generate [-w] [-k K]

# Examples
    parallel ./generate.py -w -k {}  ::: $(echo 5 10 15 20)
'''

def_conf = dict(
    save_plot = False,
    do           = 'homo',
)
def_conf.update(argparser.generate(USAGE))

####################################################
### Config
spec = _spec.EXPE_ICDM_R
spec['debug'] = 'hyper101'
#spec = _spec.EXPE_ALL_3_IBP

def exception_config(config):
    if config['model'] in ('mmsb_cgs', 'immsb'):
        if config['hyper'] == 'fix':
            return False
    if config['model'] in ('ibp_cgs', 'ibp'):
        if config['hyper'] == 'auto':
            return False
    return True

def_conf.update( {'load_data':False, # Need to compute feature and communities ground truth (no stored in pickle)
            'load_model': True, # Load model vs Gnerate random data
           } )
configs = make_forest_conf(spec)

for config in configs:
    config.update(def_conf)

    test = exception_config(config)
    if not test:
        continue

    # Generate Data source
    #frontend = FrontendManager.get(config)
    frontend = frontendNetwork(config)
    data = frontend.load_data(randomize=False)
    data = frontend.sample()
    model = ModelManager(config=config)

    # Generate Data model
    if config.get('load_model'):
        ### Generate data from a fitted model
        model = model.load()
        if model is None:
            continue
    else:
        ### Generate data from a un-fitted model
        model = model.model

    ###############################################################
    ### Expe Wrap up debug
    print frontend.output_path
    print('corpus: %s, model: %s, K = %s, N =  %s' % (frontend.corpus_name, config['model'], config['K'], config['N']) )

    if config['do'] == 'homo':
        d = {}
        #d['homo_dot_o'], d['homo_dot_e'] = frontend.homophily(model=model, sim='dot')
        #diff2 = d['homo_dot_o'] - d['homo_dot_e']
        #d['homo_model_o'], d['homo_model_e'] = frontend.homophily(model=model, sim='model')
        #diff1 = d['homo_model_o'] - d['homo_model_e']
        #homo_text =  '''Similarity | Hobs | Hexp | diff\
        #               \nmodel   %.4f  %.4f %.4f\
        #               \ndot     %.4f  %.4f %.4f\
        #''' % ( d['homo_model_o'], d['homo_model_e'] ,diff1,
        #       d['homo_dot_o'], d['homo_dot_e'], diff2)
        #print homo_text
        theta, phi = model.reduce_latent()
        K = theta.shape[1]
        d['K'] = K

        if config.get('save_plot'):
            try:
                frontend.update_json(d)
            except Exception as e:
                print e
                pass

