###############################################################################
#				Run Code for Mixed Membership Block Model
#				Authors: Alex Burnap, Efren Cruz, Xin Rong, Brian Segal
#				Date: April 1, 2013
###############################################################################

#---------------------------- Dependencies -----------------------------------#
import MMSB.MMSB_model as model
import pymc
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from frontend import frontendNetwork

#---------------------------- Run Time Params --------------------------------#

# Probably going to try and use - flag conventions with __init__(self, *args, **kwargs)


#---------------------------- Load Data --------------------------------------#
#data_matrix=np.loadtxt("../data/Y_alpha0.1_K5_N20.txt", delimiter=',')
N = 100; K = 4
data = getClique(N, K=K)
G = nx.from_numpy_matrix(data, nx.DiGraph())
data = nx.adjacency_matrix(G, np.random.permutation(range(N))).A

num_people = N
num_groups = 4
alpha = np.ones(num_groups).ravel()*0.1
#B = np.eye(num_groups)*0.85
#B = B + np.random.random(size=[num_groups,num_groups])*0.1

B = np.eye(num_groups)*0.8
B = B + np.ones([num_groups,num_groups])*0.2-np.eye(num_groups)*0.2

#---------------------------- Setup Model -----------------------------------#
raw_model = model.create_model(data, num_people, num_groups, alpha, B)
#model_instance = pymc.Model(raw_model)

#---------------------------- Call MAP to initialize MCMC -------------------#
#pymc.MAP(model_instance).fit(method='fmin_powell')
print('---------- Finished Running MAP to Set MCMC Initial Values ----------')
#---------------------------- Run MCMC --------------------------------------#
print('--------------------------- Starting MCMC ---------------------------')
M = pymc.MCMC(raw_model, db='pickle', dbname='Disaster.pickle')
M.sample(10,2, thin=2, verbose=0)

