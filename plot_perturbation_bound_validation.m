function plot_perturbation_bound_validation() 
clc;clear all;close all 
n = 32;
problem_name = 'shaw';  
[A, b_exact, x_true] = generate_test_problem(problem_name, n);
lambda = 1e-3;
maxit = n;
tol = 1e-6;
rng(0); 
B_unpert = A';
E = 1e-4 * randn(size(A));  
B_pert = B_unpert + E';           
DeltaM_AB = A * E;
DeltaM_BA = E * A; 

[~,~,~,~,~,~, phi_hba_u, dphi_hba_bound] = BAgmres_hybrid_bounds(A, B_unpert, b_exact, x_true, tol, maxit, lambda, DeltaM_BA);
[~,~,~,~,~,~, phi_hba_p, ~]              = BAgmres_hybrid_bounds(A, B_pert,   b_exact, x_true, tol, maxit, lambda, zeros(size(DeltaM_BA)));

[~,~,~,~,~,~, phi_hab_u, dphi_hab_bound] = ABgmres_hybrid_bounds(A, B_unpert, b_exact, x_true, tol, maxit, lambda, DeltaM_AB);
[~,~,~,~,~,~, phi_hab_p, ~]              = ABgmres_hybrid_bounds(A, B_pert,   b_exact, x_true, tol, maxit, lambda, zeros(size(DeltaM_AB)));

figure('Name', 'Perturbation Bound Validation', 'Position', [100 100 900 700]);
t = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, 'Validation of Perturbation Bounds at Final Iteration', 'FontSize', 14, 'FontWeight', 'bold');

plot_single_bound(nexttile, phi_hab_u, phi_hab_p, dphi_hab_bound, 'hybrid AB-GMRES');
plot_single_bound(nexttile, phi_hba_u, phi_hba_p, dphi_hba_bound, 'hybrid BA-GMRES');

end

function plot_single_bound(ax, phi_u, phi_p, dphi_bound, plot_title)
     
    k_actual = min([length(phi_u), length(phi_p), length(dphi_bound)]);
     
    if k_actual == 0
        title(ax, [plot_title ' (No iterations completed)']);
        return;
    end
     
    phi_p_k = phi_p{k_actual};
    phi_u_k = phi_u{k_actual};
    dphi_bound_k = dphi_bound{k_actual}; 
    actual_change = abs(phi_p_k - phi_u_k);
    theoretical_bound = abs(dphi_bound_k);
     
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
