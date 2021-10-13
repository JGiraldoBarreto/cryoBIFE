/**
 * AUTHORS
 * Julian Giraldo-Barreto, Pilar Cossio, Alex Barnett, Bob Carpenter
 * 
 * REFERENCES
 * Julian Giraldo-Barreto, Sebastian Ortiz, Erik H. Thiede, Karen
 * Palacio-Rodriguez, Bob Carpenter, Alex H. Barnett & Pilar Cossio. 2021.
 * A Bayesian approach to extracting free-energy profiles from
 * cryo-electron microscopy experiments.  Scientific Reports 11.  
 * http://dx.doi.org/10.1038/s41598-021-92621-1
 *
 * COPYRIGHT
 * Simons Foundation, Julian Giraldo-Barreto
 *
 * RELEASED UNDER LICENSE
 * BSD-3
 */
functions {
  vector diff(vector a) {
    int N = rows(a);
    return a[2:N] - a[1:N - 1];
  }
}
data {
  int<lower=0> M;
  int<lower=0> N;
  matrix<lower=0, upper=1>[N, M] Pmat;
}
parameters {
  vector[M - 1] G;
}
transformed parameters {
  simplex[M] rho = softmax(-append_row(G, 0));
}
model {
  // prior
  target += -2 * log(sum(diff(G)^2));

  // likelihood
  target += sum(log(Pmat * rho));
}
