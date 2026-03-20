% run_pf.m — Run Dynare perfect foresight solver and extract results
%
% Expected inputs (set before calling):
%   mod_file     : string, name of .mod file (e.g., 'credibility_pf.mod')
%
% The shock configuration is expected to be inside the .mod file.
% To change shock size/periods, modify the .mod file before calling.
%
% Output: saves to output/perfect_foresight.mat

addpath('~/home/dynare/dynare7/lib/dynare/matlab/');

% Run Dynare (includes perfect_foresight_setup and perfect_foresight_solver)
eval(['dynare ' mod_file ' noclearall']);

% oo_.endo_simul has dimensions (nvar x (periods + 2))
% Column 1 = initial steady state
% Column 2..end-1 = simulation periods
% Column end = terminal steady state

nvar = M_.endo_nbr;
var_names = M_.endo_names;

% Extract paths (skip initial and terminal SS columns)
n_periods = size(oo_.endo_simul, 2) - 2;
endo_path = oo_.endo_simul(:, 2:end-1)';  % (n_periods x nvar)

% Save individual variable paths
pf_data = struct();
for k = 1:nvar
    vname = strtrim(var_names{k});
    pf_data.(vname) = endo_path(:, k);
end

% Save
if ~exist('output', 'dir'); mkdir('output'); end
save('output/perfect_foresight.mat', 'pf_data', 'var_names', 'n_periods');
disp('=== Perfect foresight results saved ===');
