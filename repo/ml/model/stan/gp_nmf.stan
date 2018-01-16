data {
	int<lower=0> U;
	int<lower=0> I;
	int<lower=0> K;
	int<lower=0> y[U,I];
	real <lower=0> a;
	real <lower=0> b;
	real <lower=0> c;
	real <lower=0> d;
}
parameters {
	positive_ordered[K] theta[U]; // user preference
	vector <lower=0>[K] beta[I];
	// item attributes
}
model {
	for (u in 1:U)
		theta[u] ~ gamma(a, b); // componentwise gamma
	for (i in 1:I)
		beta[i] ~ gamma(c, d); // componentwise gamma
	for (u in 1:U) {
		for (i in 1:I) {
			y[u,i] ~ poisson(theta[u]‘*beta[i]);
		}
	}
