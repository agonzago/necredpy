// five_sector_est.mod -- 5-sector QPM for estimation (8 obs, 8 shocks)
//
// Based on five_sector_network.mod but configured for estimation:
//   - TFP shocks REMOVED (a_j are deterministic from SS)
//   - Measurement error eps_agg added to CPI aggregation
//   - 8 shocks: eps_y, eps_m, eps_pi1..eps_pi5, eps_agg
//   - 8 observables: y, i_nom, pi_agg, pi1..pi5
//
// The inversion filter uses partial observation:
//   Given 8 observed variables and the model, it recovers 8 shocks
//   and reconstructs the full 27-variable state each period.

var y i_nom pi_agg
    pi1 pi2 pi3 pi4 pi5
    mc1 mc2 mc3 mc4 mc5
    relp1 relp2 relp3 relp4
    a1 a2 a3 a4 a5
    cp1 cp2 cp3 cp4 cp5;

// 8 shocks for 8 observables
varexo eps_y eps_m eps_agg
       eps_pi1 eps_pi2 eps_pi3 eps_pi4 eps_pi5;

parameters betta siggma vphi
           s1 s2 s3 s4 s5
           omega_12 omega_13 omega_14 omega_15
           omega_21 omega_23 omega_24 omega_25
           omega_31 omega_32 omega_34 omega_35
           omega_41 omega_42 omega_43 omega_45
           omega_51 omega_52 omega_53 omega_54
           alpha_VA1 alpha_VA2 alpha_VA3 alpha_VA4 alpha_VA5
           kappa1 kappa2 kappa3 kappa4 kappa5
           lambda1 lambda2 lambda3 lambda4 lambda5
           omega
           rho_y phi_pi phi_y rho_i
           rho_a1 rho_a2 rho_a3 rho_a4 rho_a5
           rho_cp1 rho_cp2 rho_cp3 rho_cp4 rho_cp5
           omega_H omega_L epsilon_bar
           delta_up delta_down cred_threshold;

// =====================================================
// CALIBRATION (same as five_sector_network.mod)
// =====================================================

betta  = 0.99;
siggma = 1.0;
vphi   = 1.0;

s1 = 0.10;  s2 = 0.15;  s3 = 0.10;  s4 = 0.25;  s5 = 0.40;

omega_12 = 0.02;  omega_13 = 0.05;  omega_14 = 0.03;  omega_15 = 0.02;
omega_21 = 0.08;  omega_23 = 0.04;  omega_24 = 0.05;  omega_25 = 0.03;
omega_31 = 0.25;  omega_32 = 0.02;  omega_34 = 0.05;  omega_35 = 0.03;
omega_41 = 0.10;  omega_42 = 0.05;  omega_43 = 0.08;  omega_45 = 0.07;
omega_51 = 0.05;  omega_52 = 0.03;  omega_53 = 0.05;  omega_54 = 0.07;

alpha_VA1 = 0.88;  alpha_VA2 = 0.80;  alpha_VA3 = 0.65;
alpha_VA4 = 0.70;  alpha_VA5 = 0.80;

kappa1 = 0.25;  kappa2 = 0.15;  kappa3 = 0.12;
kappa4 = 0.08;  kappa5 = 0.04;

lambda1 = 0.80;  lambda2 = 0.70;  lambda3 = 0.65;
lambda4 = 0.60;  lambda5 = 0.50;
omega   = 0.65;

rho_y  = 0.60;  phi_pi = 2.5;  phi_y = 0.5;  rho_i = 0.75;

rho_a1 = 0.85;  rho_a2 = 0.80;  rho_a3 = 0.90;
rho_a4 = 0.90;  rho_a5 = 0.90;

rho_cp1 = 0.85;  rho_cp2 = 0.30;  rho_cp3 = 0.70;
rho_cp4 = 0.50;  rho_cp5 = 0.40;

omega_H = 0.65;  omega_L = 0.35;
epsilon_bar = 2.0;
delta_up = 0.05;  delta_down = 0.70;  cred_threshold = 0.5;


// =====================================================
// MODEL
// =====================================================

model(linear);

    // ----- IS curve -----
    y = rho_y*y(-1) - (1/siggma)*(i_nom - pi_agg(+1)) + eps_y;

    // ----- Taylor rule -----
    i_nom = rho_i*i_nom(-1) + (1-rho_i)*(phi_pi*pi_agg + phi_y*y) + eps_m;

    // ----- CPI aggregation + measurement error -----
    // eps_agg absorbs discrepancy between model weights and data aggregation
    pi_agg = s1*pi1 + s2*pi2 + s3*pi3 + s4*pi4 + s5*pi5 + eps_agg;

    // ----- Sectoral Phillips curves -----
    pi1 = omega*lambda1*pi1(+1) + (1-omega*lambda1)*pi1(-1) + kappa1*mc1 + cp1;
    pi2 = omega*lambda2*pi2(+1) + (1-omega*lambda2)*pi2(-1) + kappa2*mc2 + cp2;
    pi3 = omega*lambda3*pi3(+1) + (1-omega*lambda3)*pi3(-1) + kappa3*mc3 + cp3;
    pi4 = omega*lambda4*pi4(+1) + (1-omega*lambda4)*pi4(-1) + kappa4*mc4 + cp4;
    pi5 = omega*lambda5*pi5(+1) + (1-omega*lambda5)*pi5(-1) + kappa5*mc5 + cp5;

    // ----- Marginal costs (static, network linkages) -----
    mc1 = alpha_VA1*(siggma+vphi)*y
        + omega_12*relp2 + omega_13*relp3 + omega_14*relp4
        + omega_15*(-(s1*relp1 + s2*relp2 + s3*relp3 + s4*relp4)/s5)
        - relp1 - a1;

    mc2 = alpha_VA2*(siggma+vphi)*y
        + omega_21*relp1 + omega_23*relp3 + omega_24*relp4
        + omega_25*(-(s1*relp1 + s2*relp2 + s3*relp3 + s4*relp4)/s5)
        - relp2 - a2;

    mc3 = alpha_VA3*(siggma+vphi)*y
        + omega_31*relp1 + omega_32*relp2 + omega_34*relp4
        + omega_35*(-(s1*relp1 + s2*relp2 + s3*relp3 + s4*relp4)/s5)
        - relp3 - a3;

    mc4 = alpha_VA4*(siggma+vphi)*y
        + omega_41*relp1 + omega_42*relp2 + omega_43*relp3
        + omega_45*(-(s1*relp1 + s2*relp2 + s3*relp3 + s4*relp4)/s5)
        - relp4 - a4;

    mc5 = alpha_VA5*(siggma+vphi)*y
        + omega_51*relp1 + omega_52*relp2 + omega_53*relp3 + omega_54*relp4
        + 0*(-(s1*relp1 + s2*relp2 + s3*relp3 + s4*relp4)/s5)
        - (-(s1*relp1 + s2*relp2 + s3*relp3 + s4*relp4)/s5)
        - a5;

    // ----- Relative price identities -----
    relp1 = relp1(-1) + pi1 - pi_agg;
    relp2 = relp2(-1) + pi2 - pi_agg;
    relp3 = relp3(-1) + pi3 - pi_agg;
    relp4 = relp4(-1) + pi4 - pi_agg;

    // ----- TFP processes (deterministic, no shocks) -----
    a1 = rho_a1*a1(-1);
    a2 = rho_a2*a2(-1);
    a3 = rho_a3*a3(-1);
    a4 = rho_a4*a4(-1);
    a5 = rho_a5*a5(-1);

    // ----- Cost-push processes -----
    cp1 = rho_cp1*cp1(-1) + eps_pi1;
    cp2 = rho_cp2*cp2(-1) + eps_pi2;
    cp3 = rho_cp3*cp3(-1) + eps_pi3;
    cp4 = rho_cp4*cp4(-1) + eps_pi4;
    cp5 = rho_cp5*cp5(-1) + eps_pi5;

end;

regimes;
  M1: omega = omega_H;
  M2: omega = omega_L;

  switch: shadow_cred;
  monitor: pi_agg;
  band: epsilon_bar;
  cred_threshold: cred_threshold;
  delta_up: delta_up;
  delta_down: delta_down;
  cred_init: 1.0;
end;

// Priors for estimation
// Observables: y, i_nom, pi_agg, pi1, pi2, pi3, pi4, pi5
priors;
  // Phillips curve slopes
  kappa1,   normal,      0.25, 0.10, 0.01, 1.0;
  kappa2,   normal,      0.15, 0.08, 0.01, 1.0;
  kappa3,   normal,      0.12, 0.06, 0.01, 1.0;
  kappa4,   normal,      0.08, 0.04, 0.01, 0.5;
  kappa5,   normal,      0.04, 0.02, 0.01, 0.3;

  // Taylor rule
  phi_pi,   normal,      2.5,  0.5,  1.01, 5.0;
  phi_y,    normal,      0.5,  0.2,  0.01, 2.0;
  rho_i,    beta_dist,   0.75, 0.10;

  // Shock standard deviations (8 shocks)
  sigma_eps_y,     inv_gamma,  0.5,  0.3;
  sigma_eps_m,     inv_gamma,  0.25, 0.15;
  sigma_eps_agg,   inv_gamma,  0.10, 0.05;
  sigma_eps_pi1,   inv_gamma,  1.0,  0.5;
  sigma_eps_pi2,   inv_gamma,  1.0,  0.5;
  sigma_eps_pi3,   inv_gamma,  0.5,  0.3;
  sigma_eps_pi4,   inv_gamma,  0.5,  0.3;
  sigma_eps_pi5,   inv_gamma,  0.3,  0.2;
end;
