// credibility_nk_banrep.mod -- 3-equation NK model with BanRep credibility
//
// Same model as credibility_nk_regimes.mod but uses a credibility; block
// with BanRep Gaussian signal and AR(1) accumulation.
//
// State vector: [y, pi, ii]

var y pi ii;
varexo eps_d eps_s eps_m;

parameters beta sigma kappa rho_i phi_pi phi_y omega
           omega_H omega_L epsilon_bar
           delta_up delta_down cred_threshold
           psi_cred omega_sig_ub omega_sig1 omega_sig3;

beta    = 0.99;
sigma   = 1.0;
kappa   = 0.3;
rho_i   = 0.7;
phi_pi  = 1.5;
phi_y   = 0.5;
omega   = 0.65;

// Credibility parameters
omega_H        = 0.65;
omega_L        = 0.35;
epsilon_bar    = 2.0;
delta_up       = 0.05;
delta_down     = 0.70;
cred_threshold = 0.5;

// BanRep credibility parameters
psi_cred    = 0.84;
omega_sig_ub = 0.041;
omega_sig1  = 0.041;
omega_sig3  = 0.66;

model;
  // IS curve
  y = y(+1) - sigma*(ii - pi(+1)) + eps_d;
  // Phillips curve with credibility weight
  pi = omega*beta*pi(+1) + (1-omega)*pi(-1) + kappa*y + eps_s;
  // Taylor rule
  ii = rho_i*ii(-1) + (1-rho_i)*(phi_pi*pi + phi_y*y) + eps_m;
end;

regimes;
  M1: omega = omega_H;
  M2: omega = omega_L;

  switch: shadow_cred;
  monitor: pi;
  band: epsilon_bar;
  cred_threshold: cred_threshold;
  delta_up: delta_up;
  delta_down: delta_down;
  cred_init: 1.0;
end;

credibility;
  monitor:   pi;
  threshold: 0.5;
  cred_init: 1.0;
  signal = exp(-omega_sig_ub - omega_sig1*(pi(-1) - pi_star)^2) - omega_sig3;
  accumulation = psi_cred*cred + (1-psi_cred)*s;
end;

priors;
  kappa,    normal,      0.3,  0.15, 0.01, 1.0;
  phi_pi,   normal,      1.5,  0.3,  1.01, 3.0;
  phi_y,    normal,      0.5,  0.2,  0.01, 2.0;
  rho_i,    beta_dist,   0.7,  0.1;

  sigma_d,  inv_gamma,   0.5,  0.3;
  sigma_s,  inv_gamma,   0.5,  0.3;
  sigma_m,  inv_gamma,   0.25, 0.15;
end;
