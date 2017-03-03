/** 
 * Mixed Membership Stochastic Blockmodel
 */
data {
  int<lower=1> K; // number of communities
  int<lower=2> N; // number of vertices
  //matrix<lower=0, upper=1>[N,N] Ew; // weighted edges, shifted to [0, 1]
  int<lower=0, upper=1> En[N,N]; // unweighted binary edges

  // Hyper-parameters
  vector<lower=0>[K] alpha; // Dirichlet distribution
  matrix[K,K] Blk; // block connectivity
}

parameters {
  simplex[K] mu[N]; // membership probability per vertex
}

model {
  for (v in 1:N)
    mu[v] ~ dirichlet(alpha);
  
  for (u in 1:(N-1))
    for (v in (u+1):N) // assuming En is symmetric, ignoring the diagonals
      En[u][v] ~ bernoulli( (mu[u])' * Blk * mu[v] );
}