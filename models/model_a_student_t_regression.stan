
data {
  int<lower=1> N;
  int<lower=1> K;
  matrix[N, K] X;
  vector[N] y;
  real y_bar;
}
parameters {
  real alpha;
  vector[K] beta;
  real<lower=0> sigma;
  real<lower=0> nu_minus2;
}
transformed parameters {
  real<lower=2> nu = nu_minus2 + 2;
  vector[N] mu = alpha + X * beta;
}
model {
  alpha ~ normal(y_bar, 2);
  beta ~ normal(0, 1);
  sigma ~ exponential(1);
  nu_minus2 ~ exponential(1);
  y ~ student_t(nu, mu, sigma);
}
generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  for (n in 1:N) {
    log_lik[n] = student_t_lpdf(y[n] | nu, mu[n], sigma) - y[n];
    y_rep[n] = student_t_rng(nu, mu[n], sigma);
  }
}
