// credibility_nk_regimes.mod -- 3-equation NK model with regimes block
//
// Same model as credibility_nk.mod but uses the regimes; block syntax
// so that parse_two_regime_model() returns matrices + switching function
// in a single call.
//
// State vector: [y, pi, ii, pi_lag]

var y pi ii pi_lag;
varexo eps_d eps_s eps_m;

parameters beta sigma kappa rho_i phi_pi phi_y omega
           omega_H omega_L epsilon_bar
           delta_up delta_down cred_threshold;

beta    = 0.99;
sigma   = 1.0;
kappa   = 0.3;
rho_i   = 0.7;
phi_pi  = 1.5;
phi_y   = 0.5;
omega   = 0.65;

// Credibility parameters (used by regimes block)
omega_H        = 0.65;
omega_L        = 0.35;
epsilon_bar    = 2.0;
delta_up       = 0.05;
delta_down     = 0.70;
cred_threshold = 0.5;

model;
  // IS curve
  y = y(+1) - sigma*(ii - pi(+1)) + eps_d;
  // Phillips curve with credibility weight
  pi = omega*beta*pi(+1) + (1-omega)*pi_lag + kappa*y + eps_s;
  // Taylor rule
  ii = rho_i*ii(-1) + (1-rho_i)*(phi_pi*pi + phi_y*y) + eps_m;
  // Identity: lagged inflation
  pi_lag = pi(-1);
end;

regimes;
  M1: omega = omega_H;
  M2: omega = omega_L;

  // Switching parameters reference declared parameters (estimable)
  switch: shadow_cred;
  monitor: pi;
  band: epsilon_bar;
  cred_threshold: cred_threshold;
  delta_up: delta_up;
  delta_down: delta_down;
  cred_init: 1.0;
end;

// Priors for Bayesian estimation (Dynare-inspired syntax)
// Format: name, distribution, p1, p2 [, lower_bound, upper_bound];
//
// Distributions: normal, beta_dist, gamma_dist, inv_gamma,
//                uniform, half_normal
//
// Parameters NOT listed here are held fixed at their calibrated values.

priors;
  // Structural parameters
  // Format: name, distribution, mean, std [, lower_bound, upper_bound];
  //
  // For beta_dist, gamma_dist, inv_gamma: (mean, std) are converted to
  // shape parameters internally. This prevents U-shaped beta priors.
  kappa,    normal,      0.3,  0.15, 0.01, 1.0;    // Phillips curve slope
  phi_pi,   normal,      1.5,  0.3,  1.01, 3.0;    // Taylor rule: inflation
  phi_y,    normal,      0.5,  0.2,  0.01, 2.0;    // Taylor rule: output gap
  rho_i,    beta_dist,   0.7,  0.1;                 // mean=0.7, std=0.1

  // Shock standard deviations
  sigma_d,  inv_gamma,   0.5,  0.3;                 // mean=0.5, std=0.3
  sigma_s,  inv_gamma,   0.5,  0.3;                 // mean=0.5, std=0.3
  sigma_m,  inv_gamma,   0.25, 0.15;                // mean=0.25, std=0.15
end;
