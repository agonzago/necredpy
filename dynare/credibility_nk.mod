// credibility_nk.mod — 3-equation New Keynesian model
// Used by: (1) our parser for Pontus solver, (2) Dynare stoch_simul for linear IRFs
//
// State vector: [y, pi, ii, pi_lag]
// omega is a parameter — evaluate at omega_H or omega_L for regime switching

var y pi ii pi_lag;
varexo eps_d eps_s eps_m;

parameters beta sigma kappa rho_i phi_pi phi_y omega;

beta    = 0.99;
sigma   = 1.0;
kappa   = 0.3;
rho_i   = 0.7;
phi_pi  = 1.5;
phi_y   = 0.5;
omega   = 0.65;

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

shocks;
  var eps_d; stderr 0.5;
  var eps_s; stderr 0.5;
  var eps_m; stderr 0.25;
end;

stoch_simul(order=1, irf=0, noprint, nograph);
