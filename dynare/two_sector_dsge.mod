%------------------------------------------------------------------
% Two-Sector NK DSGE with Input-Output Linkages
% STRUCTURAL VERSION -- FULL MARGINAL COST
%
% The mc equations show the own-relative-price term (-relp_j)
% explicitly. Its coefficient is EXACTLY 1 by CRS.
% This is the structural error-correction mechanism.
%
% Author: Andres Gonzalez (IMF), March 2026
%------------------------------------------------------------------

var 
    y i_nom pi_agg pi1 pi2 mc1 mc2 relp a1 a2;

varexo eps_a1 eps_a2 eps_m;

parameters 
    betta siggma vphi
    alpha_VA1 alpha_VA2 omega_12 omega_21
    s1 s2 kappa1 kappa2
    phi_pi phi_y rho_i rho_a1 rho_a2;

betta=0.99; siggma=1.0; vphi=1.0;
omega_21=0.25; omega_12=0.05;
alpha_VA1=1-omega_12; alpha_VA2=1-omega_21;
s1=0.20; s2=1-s1;
kappa1=0.15; kappa2=0.03;
phi_pi=1.5; phi_y=0.5; rho_i=0.75;
rho_a1=0.90; rho_a2=0.90;

%------------------------------------------------------------------
% MARGINAL COST DERIVATION
%
% mc_j = alpha_VA_j*(w-p_j) + sum_k omega_jk*(p_k-p_j) - a_j
%
% Expanding:
%   mc_j = alpha_VA_j*(sigma+phi)*y + sum_k omega_jk*relp_k
%          - (alpha_VA_j + sum_k omega_jk)*relp_j - a_j
%
% By CRS (closed economy): alpha_VA_j + sum_k omega_jk = 1.
% Therefore:
%
%   mc_j = alpha_VA_j*(sigma+phi)*y + sum_k omega_jk*relp_k
%          - 1*relp_j - a_j
%           ^^^^^^^^
%     THIS IS THE STRUCTURAL ERROR CORRECTION. Coefficient = 1.
%     Not a free parameter. Not ad hoc. It's CRS.
%
% With relp1=relp, relp2=-(s1/s2)*relp:
%------------------------------------------------------------------

model(linear);

    % IS curve
    y = y(+1) - (1/siggma)*(i_nom - pi_agg(+1));

    % NKPC
    pi1 = betta*pi1(+1) + kappa1*mc1;
    pi2 = betta*pi2(+1) + kappa2*mc2;

    % MC sector 1: mc1 = alpha_VA1*(sig+phi)*y + omega_12*relp2 - relp1 - a1
    mc1 = alpha_VA1*(siggma+vphi)*y 
        - omega_12*(s1/s2)*relp    % network term (omega_12*relp2)
        - relp                     % OWN-PRICE: -1*relp1 (CRS)
        - a1;

    % MC sector 2: mc2 = alpha_VA2*(sig+phi)*y + omega_21*relp1 - relp2 - a2
    mc2 = alpha_VA2*(siggma+vphi)*y 
        + omega_21*relp            % NETWORK: omega_21*relp1
        + (s1/s2)*relp             % OWN-PRICE: -relp2 = (s1/s2)*relp (CRS)
        - a2;

    % Aggregation
    pi_agg = s1*pi1 + s2*pi2;
    relp = relp(-1) + pi1 - pi_agg;

    % Policy
    i_nom = rho_i*i_nom(-1) + (1-rho_i)*(phi_pi*pi_agg + phi_y*y) + eps_m;

    % Exogenous
    a1 = rho_a1*a1(-1) + eps_a1;
    a2 = rho_a2*a2(-1) + eps_a2;

end;

initval;
y=0;i_nom=0;pi_agg=0;pi1=0;pi2=0;mc1=0;mc2=0;relp=0;a1=0;a2=0;
end;
steady; check;

shocks;
var eps_a1=0.01^2; var eps_a2=0.01^2; var eps_m=0.0025^2;
end;

stoch_simul(order=1, irf=40, nograph) y pi_agg pi1 pi2 mc1 mc2 relp i_nom;
