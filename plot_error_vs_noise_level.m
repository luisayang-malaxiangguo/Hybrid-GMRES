function plot_error_vs_noise_level()
clear all; clc; close all

%%
n = 32;
problem_name = 'shaw';
[A, b_exact, x_true] = generate_test_problem(problem_name, n);

rng(0); 
E = 1e-4 * randn(size(A'));
B_pert = A' + E;
DeltaM_AB = A * E;
DeltaM_BA = E * A;
maxit = n;
tol = 1e-6;

%% 
noise_levels = logspace(-4, -1, 20);
 
final_errors_hab = zeros(size(noise_levels)); % Hybrid AB
final_errors_hba = zeros(size(noise_levels)); % Hybrid BA
final_errors_nab = zeros(size(noise_levels)); % Non-hybrid AB
final_errors_nba = zeros(size(noise_levels)); % Non-hybrid BA

%% 
tic;

for i = 1:length(noise_levels)
    level = noise_levels(i);
    
    noise = randn(size(b_exact));
    b_noise = b_exact + (noise / norm(noise)) * level * norm(b_exact);
    
    k_gcv = 20;
    m = size(A,1);
    options = optimset('Display', 'off', 'TolX', 1e-8);

    gcv_handle_ab = @(lambda) gcv_function(lambda, A, B_pert, b_noise, m, k_gcv, 'ab');
    lambda_gcv_ab = fminbnd(gcv_handle_ab, 1e-9, 1e-1, options);
    [~, err_hab, ~, ~] = ABgmres_hybrid_bounds(A, B_pert, b_noise, x_true, tol, maxit, lambda_gcv_ab, DeltaM_AB);
    final_errors_hab(i) = err_hab(end);
    
    gcv_handle_ba = @(lambda) gcv_function(lambda, A, B_pert, b_noise, m, k_gcv, 'ba');
    lambda_gcv_ba = fminbnd(gcv_handle_ba, 1e-9, 1e-1, options);
    [~, err_hba, ~, ~] = BAgmres_hybrid_bounds(A, B_pert, b_noise, x_true, tol, maxit, lambda_gcv_ba, DeltaM_BA);
    final_errors_hba(i) = err_hba(end);
     
    [~, err_nab, ~, ~] = ABgmres_nonhybrid_bounds(A, B_pert, b_noise, x_true, tol, maxit, DeltaM_AB);
    final_errors_nab(i) = err_nab(end);
    
    [~, err_nba, ~, ~] = BAgmres_nonhybrid_bounds(A, B_pert, b_noise, x_true, tol, maxit, DeltaM_BA);
    final_errors_nba(i) = err_nba(end);
    
end
toc;
 
figure('Name', 'Error vs. Noise Level (All Methods)', 'Position', [300 300 800 600]);

loglog(noise_levels, final_errors_hab, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Hybrid AB');
hold on;
loglog(noise_levels, final_errors_hba, 'r-x', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Hybrid BA');

loglog(noise_levels, final_errors_nab, 'b--s', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Non-Hybrid AB');
loglog(noise_levels, final_errors_nba, 'r--d', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Non-Hybrid BA');
hold off;

grid on;
title('Final Solution Error vs. Data Noise Level');
xlabel('Relative Noise Level ||e|| / ||b_{exact}||');
ylabel('Final Relative Error ||x_k - x_{true}|| / ||x_{true}||');
legend('show', 'Location', 'NorthWest');
axis tight;
ylim([min(final_errors_hab)*0.8, max(final_errors_nba)*1.2]); % Adjust y-axis for better view
set(gca, 'FontSize', 12);
end

