// two_sector_network.mod -- Two-sector QPM with I-O linkages (linear, no credibility)
//
// Baseline network model BEFORE adding credibility mechanism.
// Solves with both Dynare stoch_simul and Pontus single-regime.
//
// Sectors:
//   1 = upstream (tradable, energy-like): high kappa, high import share
//   2 = downstream (non-tradable, services-like): low kappa, low import share
//
// Network linkage: sector 2 uses sector 1 output as intermediate input
//   omega_21 = 0.25 (downstream buys from upstream)
//   omega_12 = 0.05 (upstream buys little from downstream)
//
// Equations:
//   IS:   y = rho_y*y(-1) - (1/sigma)*(i - E[pi_agg(+1)]) + eps_y
//   PC1:  pi1 = lambda1*E[pi1(+1)] + (1-lambda1)*pi1(-1) + kappa1*mc1 + eps_pi1
//   PC2:  pi2 = lambda2*E[pi2(+1)] + (1-lambda2)*pi2(-1) + kappa2*mc2 + eps_pi2
//   MC1:  mc1 = alpha_VA1*(sigma+vphi)*y - omega_12*(s1/s2)*relp - relp - a1
//   MC2:  mc2 = alpha_VA2*(sigma+vphi)*y + omega_21*relp + (s1/s2)*relp - a2
//   AGG:  pi_agg = s1*pi1 + s2*pi2
//   RELP: relp = relp(-1) + pi1 - pi_agg
//   TR:   i_nom = rho_i*i(-1) + (1-rho_i)*(phi_pi*pi_agg + phi_y*y) + eps_m
//   TFP:  a1 = rho_a1*a1(-1) + eps_a1
//         a2 = rho_a2*a2(-1) + eps_a2
//
// State vector: [y, i_nom, pi_agg, pi1, pi2, mc1, mc2, relp, a1, a2]

var y i_nom pi_agg pi1 pi2 mc1 mc2 relp a1 a2;
varexo eps_a1 eps_a2 eps_m eps_pi1 eps_pi2 eps_y;

parameters betta siggma vphi
           alpha_VA1 alpha_VA2 omega_12 omega_21
           s1 s2 kappa1 kappa2
           lambda1 lambda2
           rho_y phi_pi phi_y rho_i rho_a1 rho_a2;

// Structural
betta  = 0.99;
siggma = 1.0;
vphi   = 1.0;

// I-O structure
omega_21   = 0.25;   // downstream buys 25% from upstream
omega_12   = 0.05;   // upstream buys 5% from downstream
alpha_VA1  = 0.95;   // = 1 - omega_12 (CRS)
alpha_VA2  = 0.75;   // = 1 - omega_21 (CRS)

// CPI weights
s1 = 0.20;           // upstream weight in CPI
s2 = 0.80;           // downstream weight in CPI

// Phillips curve slopes (upstream more flexible)
kappa1 = 0.15;
kappa2 = 0.03;

// Hybrid expectations weights
lambda1 = 0.70;      // upstream more forward-looking
lambda2 = 0.50;      // downstream more backward-looking

// IS curve persistence
rho_y = 0.60;

// Taylor rule
phi_pi = 2.5;
phi_y  = 0.5;
rho_i  = 0.75;

// TFP persistence
rho_a1 = 0.90;
rho_a2 = 0.90;

model(linear);

    // IS curve (QPM: backward-looking persistence)
    y = rho_y*y(-1) - (1/siggma)*(i_nom - pi_agg(+1)) + eps_y;

    // Sectoral Phillips curves (hybrid)
    pi1 = lambda1*pi1(+1) + (1-lambda1)*pi1(-1) + kappa1*mc1 + eps_pi1;
    pi2 = lambda2*pi2(+1) + (1-lambda2)*pi2(-1) + kappa2*mc2 + eps_pi2;

    // Marginal costs (full form with own-relp, structural from CRS)
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

    // Taylor rule
    i_nom = rho_i*i_nom(-1) + (1-rho_i)*(phi_pi*pi_agg + phi_y*y) + eps_m;

    // Exogenous TFP processes
    a1 = rho_a1*a1(-1) + eps_a1;
    a2 = rho_a2*a2(-1) + eps_a2;

end;

initval;
y=0; i_nom=0; pi_agg=0; pi1=0; pi2=0; mc1=0; mc2=0; relp=0; a1=0; a2=0;
end;
steady; check;

shocks;
var eps_a1  = 0.01^2;
var eps_a2  = 0.01^2;
var eps_m   = 0.0025^2;
var eps_pi1 = 0.001^2;
var eps_pi2 = 0.001^2;
var eps_y   = 0.005^2;
end;

stoch_simul(order=1, irf=40, nograph) y pi_agg pi1 pi2 mc1 mc2 relp i_nom;
