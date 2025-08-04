function plot_perturbation_bound_validation()
% PLOT_PERTURBATION_BOUND_VALIDATION Validates the theoretical perturbation bounds
% by comparing them against the actual change in filter factors.

%% 1) Set up Test Problem & Parameters
n = 32;
[A, b, x_true] = shaw(n);
lambda = 1e-3;
k_to_plot = 30; % Choose k < n 
maxit = k_to_plot;
tol = 1e-8;
rng(0); % for reproducibility

% Define unperturbed and perturbed back-projectors
B_unpert = A';
E = 1e-5 * randn(size(A)); % Create an error matrix
B_pert = B_unpert + E';             % Create the perturbed back-projector

% Define the total perturbation terms for AB and BA methods
DeltaM_AB = A * E;
DeltaM_BA = E * A;

%% 2) Run simulations for all four methods
fprintf('Running simulations for k <= %d...\n', k_to_plot);

% --- non-hybrid BA-GMRES ---
[~,~,~,~,~,~, phi_ba_u, dphi_ba_bound] = BAgmres_nonhybrid_bounds(A, B_unpert, b, x_true, tol, maxit, DeltaM_BA);
[~,~,~,~,~,~, phi_ba_p, ~]             = BAgmres_nonhybrid_bounds(A, B_pert,   b, x_true, tol, maxit, zeros(size(DeltaM_BA)));
% --- non-hybrid AB-GMRES ---
[~,~,~,~,~,~, phi_ab_u, dphi_ab_bound] = ABgmres_nonhybrid_bounds(A, B_unpert, b, x_true, tol, maxit, DeltaM_AB);
[~,~,~,~,~,~, phi_ab_p, ~]             = ABgmres_nonhybrid_bounds(A, B_pert,   b, x_true, tol, maxit, zeros(size(DeltaM_AB)));
% --- hybrid BA-GMRES ---
[~,~,~,~,~,~, phi_hba_u, dphi_hba_bound] = BAgmres_hybrid_bounds(A, B_unpert, b, x_true, tol, maxit, lambda, DeltaM_BA);
[~,~,~,~,~,~, phi_hba_p, ~]              = BAgmres_hybrid_bounds(A, B_pert,   b, x_true, tol, maxit, lambda, zeros(size(DeltaM_BA)));
% --- hybrid AB-GMRES ---
[~,~,~,~,~,~, phi_hab_u, dphi_hab_bound] = ABgmres_hybrid_bounds(A, B_unpert, b, x_true, tol, maxit, lambda, DeltaM_AB);
[~,~,~,~,~,~, phi_hab_p, ~]              = ABgmres_hybrid_bounds(A, B_pert,   b, x_true, tol, maxit, lambda, zeros(size(DeltaM_AB)));

fprintf('Simulations complete.\n');

%% 3) Plot 2x2 comparison
fprintf('Generating bound validation plot...\n');
figure('Name', 'Perturbation Bound Validation', 'Position', [100 100 900 700]);
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, 'Validation of Perturbation Bounds at Final Iteration', 'FontSize', 14, 'FontWeight', 'bold');

% --- Use a helper function for clean, robust plotting ---
plot_single_bound(nexttile, phi_ab_u, phi_ab_p, dphi_ab_bound, 'non-hybrid AB-GMRES');
plot_single_bound(nexttile, phi_ba_u, phi_ba_p, dphi_ba_bound, 'non-hybrid BA-GMRES');
plot_single_bound(nexttile, phi_hab_u, phi_hab_p, dphi_hab_bound, 'hybrid AB-GMRES');
plot_single_bound(nexttile, phi_hba_u, phi_hba_p, dphi_hba_bound, 'hybrid BA-GMRES');

end

% --- Local Helper Function for Plotting ---
function plot_single_bound(ax, phi_u, phi_p, dphi_bound, plot_title)
    
    % Determine the actual number of iterations completed
    k_actual = min([length(phi_u), length(phi_p), length(dphi_bound)]);
    
    % Check if the simulation produced any results
    if k_actual == 0
        title(ax, [plot_title ' (No iterations completed)']);
        return;
    end
    
    % Extract the data from the final iteration
    phi_p_k = phi_p{k_actual};
    phi_u_k = phi_u{k_actual};
    dphi_bound_k = dphi_bound{k_actual};
    
    % Calculate the actual change vs. the theoretical bound
    actual_change = abs(phi_p_k - phi_u_k);
    theoretical_bound = abs(dphi_bound_k);
    
    % Plot the results
    semilogy(ax, 1:k_actual, actual_change, 'o-', 'DisplayName','Actual Change |Δφ|');
    hold(ax, 'on');
    semilogy(ax, 1:k_actual, theoretical_bound, 'x--', 'DisplayName','Theoretical Bound |δφ|');
    hold(ax, 'off');
    grid(ax, 'on');
    title(ax, sprintf('%s (k=%d)', plot_title, k_actual));
    xlabel(ax, 'Mode index i');
    ylabel(ax, 'Magnitude');
    legend(ax, 'Location','Best');
end