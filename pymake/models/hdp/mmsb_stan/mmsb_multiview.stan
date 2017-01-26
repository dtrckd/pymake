/** 
 * Mixed Membership Stochastic Blockmodel
 */
data {
  int<lower=1> K; // number of communities
  int<lower=2> N; // number of vertices
  int<lower=2> M; // number of sample graphs
  int<lower=0, upper=1> En[M,N,N]; // unweighted binary edges

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

  for (i in 1:M)
    for (u in 1:(N-1))
      for (v in (u+1):N) // assuming En is symmetric, ignoring the diagonals
	En[i][u][v] ~ bernoulli( (mu[u])' * Blk * mu[v] );
}