// two_sector_qpm_regimes.mod -- Two-sector QPM with I-O linkages + credibility
//
// Combines the two-sector network QPM (from References/two_sector_qpm.mod)
// with the regimes block for endogenous credibility switching.
//
// The regime-dependent parameter is omega, which controls the forward-looking
// weight in BOTH sectoral Phillips curves:
//   lambda_j(regime) = lambda_bar_j * omega_factor
// Here we use the simpler approach: omega enters directly as the
// forward-looking weight (same as the single-sector model).
//
// State vector: [y, pi1, pi2, pi_agg, mc1, mc2, relp, i_nom, a1, a2]

var y i_nom pi_agg pi1 pi2 mc1 mc2 relp a1 a2;
varexo eps_a1 eps_a2 eps_m eps_pi1 eps_pi2 eps_y;

parameters betta siggma vphi
           alpha_VA1 alpha_VA2 omega_12 omega_21
           s1 s2 kappa1 kappa2
           lambda1 lambda2 omega
           rho_y phi_pi phi_y rho_i rho_a1 rho_a2
           omega_H omega_L epsilon_bar
           delta_up delta_down cred_threshold;

betta=0.99; siggma=1.0; vphi=1.0;
omega_21=0.25; omega_12=0.05;
alpha_VA1=0.95; alpha_VA2=0.75;
s1=0.20; s2=0.80;
kappa1=0.15; kappa2=0.03;

// Hybrid expectations weights (base values, scaled by omega)
// Effective fwd weight = omega * lambda_j
//   M1 (omega=0.65): PC1 fwd=0.45, PC2 fwd=0.33
//   M2 (omega=0.35): PC1 fwd=0.24, PC2 fwd=0.17
lambda1=0.70; lambda2=0.50;
omega=0.65;

// QPM-specific
rho_y=0.60;
phi_pi=2.5; phi_y=0.5; rho_i=0.75;
rho_a1=0.90; rho_a2=0.90;

// Credibility parameters
omega_H=0.65; omega_L=0.35;
epsilon_bar=2.0;
delta_up=0.05; delta_down=0.70; cred_threshold=0.5;

model(linear);

    // IS curve (QPM: backward-looking persistence)
    y = rho_y*y(-1) - (1/siggma)*(i_nom - pi_agg(+1)) + eps_y;

    // Sectoral Phillips curves (hybrid, regime-dependent omega)
    pi1 = omega*lambda1*pi1(+1) + (1-omega*lambda1)*pi1(-1) + kappa1*mc1 + eps_pi1;
    pi2 = omega*lambda2*pi2(+1) + (1-omega*lambda2)*pi2(-1) + kappa2*mc2 + eps_pi2;

    // Marginal costs (full form with own-relp, identical to DSGE)
    mc1 = alpha_VA1*(siggma+vphi)*y
        - omega_12*(s1/s2)*relp
        - relp
        - a1;

    mc2 = alpha_VA2*(siggma+vphi)*y
        + omega_21*relp
        + (s1/s2)*relp
        - a2;

    // Aggregation
    pi_agg = s1*pi1 + s2*pi2;
    relp = relp(-1) + pi1 - pi_agg;

    // Policy
    i_nom = rho_i*i_nom(-1) + (1-rho_i)*(phi_pi*pi_agg + phi_y*y) + eps_m;

    // Exogenous
    a1 = rho_a1*a1(-1) + eps_a1;
    a2 = rho_a2*a2(-1) + eps_a2;

end;

regimes;
  M1: omega = omega_H;
  M2: omega = omega_L;

  // Switching parameters reference declared parameters (estimable)
  switch: shadow_cred;
  monitor: pi_agg;
  band: epsilon_bar;
  cred_threshold: cred_threshold;
  delta_up: delta_up;
  delta_down: delta_down;
  cred_init: 1.0;
end;
