% run_stoch_simul.m — Run Dynare stoch_simul and simulate IRF
%
% Expected inputs (set before calling):
%   mod_file     : string, name of .mod file (e.g., 'credibility_nk')
%   shock_name   : string, name of shock (e.g., 'eps_s')
%   shock_size   : scalar, shock magnitude
%   T_irf        : integer, IRF horizon
%   param_overrides : struct with parameter name/value pairs (optional)
%
% Output: saves to output/stoch_simul_irf.mat

addpath('~/home/dynare/dynare7/lib/dynare/matlab/');

% Run Dynare
eval(['dynare ' mod_file ' noclearall']);

% Parameter overrides are handled by the Python runner writing a temp .mod file

% Find shock index
shock_idx = strmatch(shock_name, M_.exo_names, 'exact');
if isempty(shock_idx)
    error(['Shock not found: ' shock_name]);
end

% Build exogenous path: shock at period 1 only
% simult_ expects (T_sim x n_exo), returns (nvar x T_sim+1) including initial conditions
exo_path = zeros(T_irf, M_.exo_nbr);
exo_path(1, shock_idx) = shock_size;  % shock at period 1

% Simulate using Dynare's simult_ (handles DR ordering internally)
% Returns y_simul: (nvar x T_irf+1) in declaration order (levels)
y_simul = simult_(M_, options_, oo_.dr.ys, oo_.dr, exo_path, 1);

% Convert to deviations from steady state, drop initial conditions column
nvar = M_.endo_nbr;
y_irf = y_simul(:, 2:end);  % drop initial SS column -> (nvar x T_irf)
y_dev = y_irf - repmat(oo_.dr.ys, 1, T_irf);

% Extract variable paths by name
var_names = M_.endo_names;
irf_data = struct();
for k = 1:nvar
    vname = strtrim(var_names{k});
    irf_data.(vname) = y_dev(k, :)';
end

% Save
if ~exist('output', 'dir'); mkdir('output'); end
save('output/stoch_simul_irf.mat', 'irf_data', 'var_names', 'T_irf', 'shock_name', 'shock_size');
disp('=== stoch_simul IRF saved ===');
