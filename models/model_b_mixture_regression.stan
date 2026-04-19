
data {
  int<lower=1> N;
  int<lower=1> K;
  matrix[N, K] X;
  vector[N] y;
  real y_bar;
}
parameters {
  real alpha1;
  vector[K] beta1;
  real<lower=0> delta;
  real<lower=0> sigma1;
  real<lower=0> sigma2;
  real<lower=0, upper=1> w;
  real<lower=0> nu1_minus2;
  real<lower=0> nu2_minus2;
}
transformed parameters {
  real<lower=2> nu1 = nu1_minus2 + 2;
  real<lower=2> nu2 = nu2_minus2 + 2;
  vector[N] mu1 = alpha1 + X * beta1;
  vector[N] mu2 = mu1 + delta;
}
model {
  alpha1 ~ normal(y_bar, 2);
  beta1 ~ normal(0, 1);
  delta ~ exponential(1);
  sigma1 ~ exponential(1);
  sigma2 ~ exponential(1);
  w ~ beta(2, 2);
  nu1_minus2 ~ exponential(1);
  nu2_minus2 ~ exponential(1);
  for (n in 1:N) {
    target += log_mix(
      w,
      student_t_lpdf(y[n] | nu1, mu1[n], sigma1),
      student_t_lpdf(y[n] | nu2, mu2[n], sigma2)
    );
  }
}
generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  for (n in 1:N) {
    real lp1 = student_t_lpdf(y[n] | nu1, mu1[n], sigma1);
    real lp2 = student_t_lpdf(y[n] | nu2, mu2[n], sigma2);
    log_lik[n] = log_mix(w, lp1, lp2) - y[n];
    if (bernoulli_rng(w) == 1)
      y_rep[n] = student_t_rng(nu1, mu1[n], sigma1);
    else
      y_rep[n] = student_t_rng(nu2, mu2[n], sigma2);
  }
}
