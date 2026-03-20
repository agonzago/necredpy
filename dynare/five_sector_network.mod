// five_sector_network.mod -- 5-sector QPM with I-O linkages + credibility
//
// Sectors:
//   1: Energy     -- upstream, flexible prices, import-intensive
//   2: Food       -- upstream, volatile, weather-exposed
//   3: Transport  -- central node, connects energy to everything
//   4: Goods/Mfg  -- tradable, import-intensive
//   5: Services   -- downstream, sticky, largest CPI weight
//
// The I-O matrix Omega captures intermediate input flows:
//   omega_jk = cost share of sector k's output in sector j's production
//
// Illustrative I-O matrix (rows = buyer, cols = seller):
//
//        Energy  Food  Transport  Goods  Services
// Energy   --    0.02    0.05    0.03    0.02     => gamma_1 = 0.12
// Food    0.08    --     0.04    0.05    0.03     => gamma_2 = 0.20
// Trans   0.25   0.02    --      0.05    0.03     => gamma_3 = 0.35
// Goods   0.10   0.05   0.08     --      0.07     => gamma_4 = 0.30
// Serv    0.05   0.03   0.05    0.07     --       => gamma_5 = 0.20
//
// CRS: alpha_VA_j = 1 - sum_k omega_jk
//
// Relative prices: relp_j = p_j - p_CPI  (only 4 independent, drop relp5)
// Identity: relp5 = -(s1*relp1 + s2*relp2 + s3*relp3 + s4*relp4) / s5
//
// Cost-push shock persistence (key for network propagation):
//   Energy: rho_cp=0.85 (oil prices stay high for quarters)
//   Food:   rho_cp=0.30 (weather normalizes quickly)
//   Others: moderate persistence
//
// State vector (27 variables):
//   y, i_nom, pi_agg,
//   pi1..pi5, mc1..mc5, relp1..relp4,
//   a1..a5, cp1..cp5

var y i_nom pi_agg
    pi1 pi2 pi3 pi4 pi5
    mc1 mc2 mc3 mc4 mc5
    relp1 relp2 relp3 relp4
    a1 a2 a3 a4 a5
    cp1 cp2 cp3 cp4 cp5;

varexo eps_y eps_m
       eps_pi1 eps_pi2 eps_pi3 eps_pi4 eps_pi5
       eps_a1 eps_a2 eps_a3 eps_a4 eps_a5;

parameters betta siggma vphi
           // CPI weights
           s1 s2 s3 s4 s5
           // I-O matrix (omega_jk = sector j buys from sector k)
           omega_12 omega_13 omega_14 omega_15
           omega_21 omega_23 omega_24 omega_25
           omega_31 omega_32 omega_34 omega_35
           omega_41 omega_42 omega_43 omega_45
           omega_51 omega_52 omega_53 omega_54
           // Value-added shares (CRS: alpha_VA_j = 1 - sum_k omega_jk)
           alpha_VA1 alpha_VA2 alpha_VA3 alpha_VA4 alpha_VA5
           // Phillips curve slopes
           kappa1 kappa2 kappa3 kappa4 kappa5
           // Hybrid expectations weights (base, scaled by omega)
           lambda1 lambda2 lambda3 lambda4 lambda5
           omega
           // QPM parameters
           rho_y phi_pi phi_y rho_i
           // TFP persistence
           rho_a1 rho_a2 rho_a3 rho_a4 rho_a5
           // Cost-push persistence
           rho_cp1 rho_cp2 rho_cp3 rho_cp4 rho_cp5
           // Credibility parameters
           omega_H omega_L epsilon_bar
           delta_up delta_down cred_threshold;

// =====================================================
// CALIBRATION
// =====================================================

betta  = 0.99;
siggma = 1.0;
vphi   = 1.0;

// CPI weights
s1 = 0.10;    // Energy
s2 = 0.15;    // Food
s3 = 0.10;    // Transport
s4 = 0.25;    // Goods/Manufacturing
s5 = 0.40;    // Services

// I-O matrix: omega_jk = sector j buys from sector k
// Row 1: Energy buys from others
omega_12 = 0.02;  omega_13 = 0.05;  omega_14 = 0.03;  omega_15 = 0.02;
// Row 2: Food buys from others
omega_21 = 0.08;  omega_23 = 0.04;  omega_24 = 0.05;  omega_25 = 0.03;
// Row 3: Transport buys from others (energy-intensive!)
omega_31 = 0.25;  omega_32 = 0.02;  omega_34 = 0.05;  omega_35 = 0.03;
// Row 4: Goods/Mfg buys from others
omega_41 = 0.10;  omega_42 = 0.05;  omega_43 = 0.08;  omega_45 = 0.07;
// Row 5: Services buys from others
omega_51 = 0.05;  omega_52 = 0.03;  omega_53 = 0.05;  omega_54 = 0.07;

// Value-added shares (CRS)
alpha_VA1 = 0.88;  // 1 - (0.02+0.05+0.03+0.02) = 0.88
alpha_VA2 = 0.80;  // 1 - (0.08+0.04+0.05+0.03) = 0.80
alpha_VA3 = 0.65;  // 1 - (0.25+0.02+0.05+0.03) = 0.65
alpha_VA4 = 0.70;  // 1 - (0.10+0.05+0.08+0.07) = 0.70
alpha_VA5 = 0.80;  // 1 - (0.05+0.03+0.05+0.07) = 0.80

// Phillips curve slopes (flexible -> sticky)
kappa1 = 0.25;    // Energy: very flexible
kappa2 = 0.15;    // Food: flexible
kappa3 = 0.12;    // Transport: moderate
kappa4 = 0.08;    // Goods: moderate-sticky
kappa5 = 0.04;    // Services: very sticky

// Hybrid expectations weights
lambda1 = 0.80;   // Energy: mostly forward-looking
lambda2 = 0.70;   // Food
lambda3 = 0.65;   // Transport
lambda4 = 0.60;   // Goods
lambda5 = 0.50;   // Services: most backward-looking
omega   = 0.65;   // Credibility scaling

// QPM parameters
rho_y  = 0.60;
phi_pi = 2.5;
phi_y  = 0.5;
rho_i  = 0.75;

// TFP persistence
rho_a1 = 0.85;  rho_a2 = 0.80;  rho_a3 = 0.90;  rho_a4 = 0.90;  rho_a5 = 0.90;

// Cost-push persistence (key asymmetry!)
rho_cp1 = 0.85;   // Energy: very persistent (oil price shocks last)
rho_cp2 = 0.30;   // Food: transitory (weather normalizes)
rho_cp3 = 0.70;   // Transport: moderate (follows energy with lag)
rho_cp4 = 0.50;   // Goods: moderate
rho_cp5 = 0.40;   // Services: low (wage pressures dissipate)

// Credibility parameters
omega_H = 0.65;  omega_L = 0.35;
epsilon_bar = 2.0;
delta_up = 0.05;  delta_down = 0.70;  cred_threshold = 0.5;


// =====================================================
// MODEL
// =====================================================

model(linear);

    // ----- IS curve (aggregate, QPM-style) -----
    y = rho_y*y(-1) - (1/siggma)*(i_nom - pi_agg(+1)) + eps_y;

    // ----- Taylor rule -----
    i_nom = rho_i*i_nom(-1) + (1-rho_i)*(phi_pi*pi_agg + phi_y*y) + eps_m;

    // ----- CPI aggregation -----
    pi_agg = s1*pi1 + s2*pi2 + s3*pi3 + s4*pi4 + s5*pi5;

    // ----- Sectoral Phillips curves (hybrid, regime-dependent omega) -----
    // Cost-push enters through persistent AR(1) process cp_j
    pi1 = omega*lambda1*pi1(+1) + (1-omega*lambda1)*pi1(-1) + kappa1*mc1 + cp1;
    pi2 = omega*lambda2*pi2(+1) + (1-omega*lambda2)*pi2(-1) + kappa2*mc2 + cp2;
    pi3 = omega*lambda3*pi3(+1) + (1-omega*lambda3)*pi3(-1) + kappa3*mc3 + cp3;
    pi4 = omega*lambda4*pi4(+1) + (1-omega*lambda4)*pi4(-1) + kappa4*mc4 + cp4;
    pi5 = omega*lambda5*pi5(+1) + (1-omega*lambda5)*pi5(-1) + kappa5*mc5 + cp5;

    // ----- Marginal costs -----
    // mc_j = alpha_VA_j*(sigma+vphi)*y
    //      + sum_k omega_jk * relp_k  (network: other sectors' prices)
    //      - relp_j                    (own-price, coeff=1 by CRS)
    //      - a_j
    //
    // relp5 is NOT a variable. Substitute using:
    //   relp5 = -(s1*relp1 + s2*relp2 + s3*relp3 + s4*relp4) / s5

    // MC1 (Energy): buys from Food, Transport, Goods, Services
    mc1 = alpha_VA1*(siggma+vphi)*y
        + omega_12*relp2 + omega_13*relp3 + omega_14*relp4
        + omega_15*(-(s1*relp1 + s2*relp2 + s3*relp3 + s4*relp4)/s5)
        - relp1
        - a1;

    // MC2 (Food): buys from Energy, Transport, Goods, Services
    mc2 = alpha_VA2*(siggma+vphi)*y
        + omega_21*relp1 + omega_23*relp3 + omega_24*relp4
        + omega_25*(-(s1*relp1 + s2*relp2 + s3*relp3 + s4*relp4)/s5)
        - relp2
        - a2;

    // MC3 (Transport): buys from Energy, Food, Goods, Services
    mc3 = alpha_VA3*(siggma+vphi)*y
        + omega_31*relp1 + omega_32*relp2 + omega_34*relp4
        + omega_35*(-(s1*relp1 + s2*relp2 + s3*relp3 + s4*relp4)/s5)
        - relp3
        - a3;

    // MC4 (Goods): buys from Energy, Food, Transport, Services
    mc4 = alpha_VA4*(siggma+vphi)*y
        + omega_41*relp1 + omega_42*relp2 + omega_43*relp3
        + omega_45*(-(s1*relp1 + s2*relp2 + s3*relp3 + s4*relp4)/s5)
        - relp4
        - a4;

    // MC5 (Services): buys from Energy, Food, Transport, Goods
    mc5 = alpha_VA5*(siggma+vphi)*y
        + omega_51*relp1 + omega_52*relp2 + omega_53*relp3 + omega_54*relp4
        + 0*(-(s1*relp1 + s2*relp2 + s3*relp3 + s4*relp4)/s5)
        - (-(s1*relp1 + s2*relp2 + s3*relp3 + s4*relp4)/s5)
        - a5;

    // ----- Relative price identities (4 independent) -----
    relp1 = relp1(-1) + pi1 - pi_agg;
    relp2 = relp2(-1) + pi2 - pi_agg;
    relp3 = relp3(-1) + pi3 - pi_agg;
    relp4 = relp4(-1) + pi4 - pi_agg;

    // ----- Exogenous TFP processes -----
    a1 = rho_a1*a1(-1) + eps_a1;
    a2 = rho_a2*a2(-1) + eps_a2;
    a3 = rho_a3*a3(-1) + eps_a3;
    a4 = rho_a4*a4(-1) + eps_a4;
    a5 = rho_a5*a5(-1) + eps_a5;

    // ----- Cost-push processes (AR(1), sector-specific persistence) -----
    // Energy shocks persistent, food transitory
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
