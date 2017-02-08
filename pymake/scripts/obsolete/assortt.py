#!/usr/bin/python -u
# -*- coding: utf-8 -*-

from util.utils import argParse
from frontend.manager import ModelManager, FrontendManager

from numpy import ma
import numpy as np
import scipy as sp
#np.set_printoptions(threshold='nan')

import logging
import os
import os.path

''' Assort / Homophily
    Obsolete'''

Corpuses += _spec.CORPUS_REAL_ICDM_1
Models = _spec.MODELS_GENERATE

##################
###### MAIN ######
##################
if __name__ == '__main__':

    config = dict(
        save_plot = False,
    )
    config.update(argparser.generate(''))


    # Initializa Model
    frontend = FrontendManager.get(config)
    data = frontend.load_data(randomize=False)
    data = frontend.sample()
    # Load model
    model = ModelManager(config=config)


    if config.get('load_model'):
        ### Generate data from a fitted model
        model = model.load()
    else:
        ### Generate data from a un-fitted model
        model = model.model

    d = frontend.assort(model)
    print d
    #frontend.update_json(d)


    ### Percentage of Homophily
#    # Source
#    sim_zeros_source = sim_source[data < 1]
#    simtest_source = np.ones((data==1).sum()) / (data < 1).sum()
#    for i, _1 in enumerate(zip(*np.where(data == 1))):
#        t = (sim_source[_1] > sim_zeros_source).sum()
#        simtest_source[i] *= t
#
#    # Learn
#    sim_zeros_learn = sim_learn[y < 1]
#    simtest_learn = np.ones((y==1).sum()) / (y < 1).sum()
#    for i, _1 in enumerate(zip(*np.where(y == 1))):
#        t = (sim_learn[_1] > sim_zeros_learn).sum()
#        simtest_learn[i] *= t
#
#    print 'Probability that link where 1sim is sup to 0sim:\n \
#            source:: mean: %f, var: %f,\n \
#            learn :: mean: %f, var: %f' % (simtest_source.mean(), simtest_source.var(), simtest_learn.mean(), simtest_learn.var())
#
    ### Plot the vector of probability
    #from plot import *
    #plt.figure()
    #plt.imshow(np.tile(simtest_source[:, np.newaxis], 100))
    #plt.title('Simtest Source')
    #plt.colorbar()

    #plt.figure()
    #plt.subplot(1,2,1)
    #plt.imshow(sim_source)
    #plt.title('Source Similarity')

    #plt.subplot(1,2,2)
    #plt.imshow(sim_learn)
    #plt.title('Learn Similarity')

    #plt.subplots_adjust(left=0.06, bottom=0.1, right=0.9, top=0.87)
    #cax = plt.axes([0.93, 0.15, 0.025, 0.7])
    #plt.colorbar(cax=cax)

    #plt.figure()
    #plt.imshow(phi)
    #plt.colorbar()



    #display(True)

