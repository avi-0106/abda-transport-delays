
data {
  int<lower=1> N;
  int<lower=1> K;
  matrix[N, K] X;
  array[N] real<lower=0> delay;
  array[N] int<lower=0, upper=1> z;
  real log_delay_pos_mean;
  real logit_positive_rate;
}
parameters {
  real alpha_z;
  vector[K] beta_z;
  real alpha_d;
  vector[K] beta_d;
  real<lower=0> sigma_d;
}
transformed parameters {
  vector[N] eta = alpha_z + X * beta_z;
  vector[N] mu_d = alpha_d + X * beta_d;
}
model {
  alpha_z ~ normal(logit_positive_rate, 1.5);
  beta_z ~ normal(0, 1);
  alpha_d ~ normal(log_delay_pos_mean, 2);
  beta_d ~ normal(0, 1);
  sigma_d ~ exponential(1);
  for (n in 1:N) {
    target += bernoulli_logit_lpmf(z[n] | eta[n]);
    if (z[n] == 1) {
      target += lognormal_lpdf(delay[n] | mu_d[n], sigma_d);
    }
  }
}
generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  for (n in 1:N) {
    if (z[n] == 1) {
      log_lik[n] = bernoulli_logit_lpmf(1 | eta[n]) + lognormal_lpdf(delay[n] | mu_d[n], sigma_d);
    } else {
      log_lik[n] = bernoulli_logit_lpmf(0 | eta[n]);
    }
    if (bernoulli_logit_rng(eta[n]) == 1) {
      real d_rep = lognormal_rng(mu_d[n], sigma_d);
      y_rep[n] = log1p(d_rep);
    } else {
      y_rep[n] = 0;
    }
  }
}
