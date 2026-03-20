// credibility_pf_v2.mod — Nonlinear PF with credibility capital stock
// Replaces the memoryless sigmoid omega_t = f(pi_t) with a dynamic
// credibility stock: cred depletes when off target, rebuilds when on target.
//
// Law of motion:
//   miss_t = sigmoid(pi_t^2 - epsilon_bar^2)
//   cred_t = cred_{t-1} + delta_up*(1-cred_{t-1})*(1-miss_t) - delta_down*cred_{t-1}*miss_t
//   omega_t = omega_L + (omega_H - omega_L)*cred_t
//
// Solved by Dynare's perfect_foresight_solver (Newton on stacked system)

var y pi ii pi_lag cred omega_t;
varexo eps_d eps_s eps_m;

parameters beta sigma kappa rho_i phi_pi phi_y;
parameters omega_H omega_L epsilon_bar steep delta_up delta_down;

beta        = 0.99;
sigma       = 1.0;
kappa       = 0.3;
rho_i       = 0.7;
phi_pi      = 1.5;
phi_y       = 0.5;
omega_H     = 0.65;
omega_L     = 0.35;
epsilon_bar = 2.0;
steep       = 50;
delta_up    = 0.05;
delta_down  = 0.7;

model;
  // Smooth miss indicator (model-local variable)
  #miss = 1 / (1 + exp(-steep*(pi*pi - epsilon_bar*epsilon_bar)));

  // Credibility capital stock (dynamic, depends on cred(-1))
  cred = cred(-1) + delta_up*(1 - cred(-1))*(1 - miss) - delta_down*cred(-1)*miss;

  // Credibility weight
  omega_t = omega_L + (omega_H - omega_L)*cred;

  // IS curve
  y = y(+1) - sigma*(ii - pi(+1)) + eps_d;

  // Phillips curve with endogenous credibility
  pi = omega_t*beta*pi(+1) + (1 - omega_t)*pi_lag + kappa*y + eps_s;

  // Taylor rule
  ii = rho_i*ii(-1) + (1 - rho_i)*(phi_pi*pi + phi_y*y) + eps_m;

  // Identity: lagged inflation
  pi_lag = pi(-1);
end;

initval;
  y = 0;
  pi = 0;
  ii = 0;
  pi_lag = 0;
  cred = 1;
  omega_t = 0.65;
  eps_d = 0;
  eps_s = 0;
  eps_m = 0;
end;

steady;

endval;
  y = 0;
  pi = 0;
  ii = 0;
  pi_lag = 0;
  cred = 1;
  omega_t = 0.65;
  eps_d = 0;
  eps_s = 0;
  eps_m = 0;
end;

steady;

// Cost-push shock at period 1 (overridden by Octave driver)
shocks;
  var eps_s;
  periods 1;
  values 3.0;
end;

perfect_foresight_setup(periods=80);
perfect_foresight_solver;
