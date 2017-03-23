import pystan
import numpy as np
import cPickle

# Hyper-parameters
K = 3 # number of communities
alpha = 0.1 * np.ones(K) # Dirichlet prior for membership mixture
block_structure = np.asarray([[0.8, 0.3, 0.2],
                              [0.2, 0.7, 0.3],
                              [0.1, 0.1, 0.6]])

# Generate M testing binary graph
M = 7
N = 10
nodes_community = np.random.randint(0, K, size = N) # values in [0, K-1]
graph_views = []
for i in xrange(M):
    binary_graph = np.diag(np.ones(N, dtype = int))
    for u in xrange(N):
        for v in xrange(u+1, N):
            u_label = nodes_community[u]
            v_label = nodes_community[v]
            edge_prob = block_structure[u_label, v_label]
            if np.random.binomial(1, edge_prob) == 1:
                binary_graph[u, v] = 1
                binary_graph[v, u] = 1

    graph_views += [binary_graph]

graph_views = np.asarray( graph_views )

# Run MCMC
def load_stan_model( model_name ):
    """
    Load stan model from disk,
    if not exist, compile the model from source code
    """
    try:
        stan_model = cPickle.load( open(model_name + ".model", 'rb') )
    except IOError:
        stan_model = pystan.StanModel( file = model_name + ".stan" )
        with open(model_name + ".model", 'wb') as fout:
            cPickle.dump(stan_model, fout)
        pass

    return stan_model



def mmsb_single_view(idx):
    mmsb_model = load_stan_model("mix_membership_blockmodel")
    fit = mmsb_model.sampling( data = {'K': K, 'N': N, 'En': graph_views[idx],
                                       'alpha': alpha, 'Blk': block_structure},
                               chains = 4 )

    # Show the result
    print(fit)
    mu = fit.extract(permuted = True)['mu']
    mean_membership = mu.mean(axis = 0)
    print(mean_membership)
    print("%d out of %d predicted right" % (sum(mean_membership.argmax(axis = 1) == nodes_community), N))


# Multiview MMSB model
mmsb_model = load_stan_model("mmsb_multiview")
fit = mmsb_model.sampling( data = {'K': K, 'N': N, 'En': graph_views,
                                   'alpha': alpha, 'Blk': block_structure} )
print(fit)
mu = fit.extract(permuted = True)['mu']
mean_membership = mu.mean(axis = 0)
print(mean_membership)
print("%d out of %d predicted right" % (sum(mean_membership.argmax(axis = 1) == nodes_community), N))
