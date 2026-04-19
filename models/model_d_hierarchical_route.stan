
data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J_route;
  matrix[N, K] X;
  array[N] int<lower=1, upper=J_route> route_id;
  vector[N] y;
}
parameters {
  real alpha;
  vector[K] beta;
  real<lower=0> sigma;
  real<lower=0> nu_minus2;
  real<lower=0> tau_route;
  vector[J_route] z_route;
}
transformed parameters {
  real<lower=2> nu = nu_minus2 + 2;
  vector[J_route] a_route = tau_route * z_route;
  vector[N] mu = alpha + X * beta;
  for (n in 1:N) {
    mu[n] += a_route[route_id[n]];
  }
}
model {
  alpha ~ normal(0, 2);
  beta ~ normal(0, 1);
  sigma ~ exponential(1);
  nu_minus2 ~ exponential(1);
  tau_route ~ exponential(1);
  z_route ~ normal(0, 1);

  y ~ student_t(nu, mu, sigma);
}
generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  for (n in 1:N) {
    log_lik[n] = student_t_lpdf(y[n] | nu, mu[n], sigma);
    y_rep[n] = student_t_rng(nu, mu[n], sigma);
  }
}
