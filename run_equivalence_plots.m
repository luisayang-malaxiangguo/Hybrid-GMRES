function run_equivalence_plots()
% This script validates the algebraic equivalences from Chapter 3 of the thesis.
clear; clc; close all;
%% 1) Set up Test Problem
fprintf('Setting up test problem...\n');
n = 32;
[A, b_exact, x_true] = generate_test_problem('shaw', n);
B = A'; % Matched-transpose case
 
rng(0);
noise = randn(size(b_exact));
b_noise = b_exact + 1e-2 * norm(b_exact) * noise / norm(noise);
maxit = n;
tol = 1e-10;
lambda = 1e-6;
%% 2) Run Solvers to Collect Data
fprintf('Running solver pairs...\n'); 
% Pair 1: BA-GMRES vs LSMR (Non-hybrid)
[x_ba, err_ba, ~, it_ba] = BAgmres_nonhybrid_bounds(A, B, b_noise, x_true, tol, maxit, zeros(size(B*A)));
[x_lsmr, err_lsmr, ~,~, it_lsmr] = lsmr_solver(A, b_noise, x_true, tol, maxit);
% Pair 2: AB-GMRES vs LSQR (Non-hybrid)
[x_ab, err_ab, ~, it_ab] = ABgmres_nonhybrid_bounds(A, B, b_noise, x_true, tol, maxit, zeros(size(A*B)));
[x_lsqr, err_lsqr, ~, it_lsqr] = lsqr_solver(A, b_noise, x_true, tol, maxit);
% Pair 3: Hybrid BA-GMRES vs Hybrid LSMR
[x_hba, err_hba, ~, it_hba] = BAgmres_hybrid_bounds(A, B, b_noise, x_true, tol, maxit, lambda, zeros(size(B*A)));
[x_hlsmr, err_hlsmr, ~, it_hlsmr] = hybrid_lsmr_solver(A, b_noise, x_true, tol, maxit, lambda);
% Pair 4: Hybrid AB-GMRES vs Hybrid LSQR (Inequivalence)
[x_hab, err_hab, ~, it_hab] = ABgmres_hybrid_bounds(A, B, b_noise, x_true, tol, maxit, lambda, zeros(size(A*B)));
[x_hlsqr, err_hlsqr, ~, it_hlsqr] = hybrid_lsqr_solver(A, b_noise, x_true, tol, maxit, lambda);
%% 3) Generate Plots
fprintf('Generating plots...\n');

% SOLUTION PLOTS 
figure('Name', 'Visual Comparison of Final Solutions', 'Position', [150, 150, 1000, 800]);
% Plot 1: BA-GMRES vs LSMR Solution
subplot(2, 2, 1);
plot(1:n, x_true, 'k-', 'LineWidth', 2.5, 'DisplayName', 'True Solution');
hold on;
plot(1:n, x_ba, 'b-', 'LineWidth', 1.5, 'DisplayName', 'BA-GMRES');
plot(1:n, x_lsmr, 'r--', 'LineWidth', 1.5, 'DisplayName', 'LSMR');
hold off; grid on;
title('BA-GMRES vs. LSMR Solution (\equiv)');
xlabel('Element Index'); ylabel('Value');
legend('show', 'Location', 'Best');
axis tight;
% Plot 2: AB-GMRES vs LSQR Solution
subplot(2, 2, 2);
plot(1:n, x_true, 'k-', 'LineWidth', 2.5, 'DisplayName', 'True Solution');
hold on;
plot(1:n, x_ab, 'b-', 'LineWidth', 1.5, 'DisplayName', 'AB-GMRES');
plot(1:n, x_lsqr, 'r--', 'LineWidth', 1.5, 'DisplayName', 'LSQR');
hold off; grid on;
title('AB-GMRES vs. LSQR Solution (\equiv)');
xlabel('Element Index');
legend('show', 'Location', 'Best');
axis tight;
% Plot 3: Hybrid BA-GMRES vs Hybrid LSMR Solution
subplot(2, 2, 3);
plot(1:n, x_true, 'k-', 'LineWidth', 2.5, 'DisplayName', 'True Solution');
hold on;
plot(1:n, x_hba, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Hybrid BA-GMRES');
plot(1:n, x_hlsmr, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Hybrid LSMR');
hold off; grid on;
title('Hybrid BA-GMRES vs. Hybrid LSMR Solution (\equiv)');
xlabel('Element Index'); ylabel('Value');
legend('show', 'Location', 'Best');
axis tight;
% Plot 4: Hybrid AB-GMRES vs Hybrid LSQR Solution
subplot(2, 2, 4);
plot(1:n, x_true, 'k-', 'LineWidth', 2.5, 'DisplayName', 'True Solution');
hold on;
plot(1:n, x_hab, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Hybrid AB-GMRES');
plot(1:n, x_hlsqr, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Hybrid LSQR');
hold off; grid on;
title('Hybrid AB-GMRES vs. Hybrid LSQR Solution (\neq)');
xlabel('Element Index');
legend('show', 'Location', 'Best');
axis tight;
sgtitle('Equivalence Validation (Final Solution)', 'FontSize', 16, 'FontWeight', 'bold');
