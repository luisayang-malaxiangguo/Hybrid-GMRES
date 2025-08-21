function run_ptr_rtp_comparison()
% This script validates the inequivalence between PTR and RTP methods from Chapter 2.

clear all; clc; close all;

%% 1) Set up Test Problem
fprintf('Setting up test problem...\n');
n = 32;
[A, b_exact, x_true] = generate_test_problem('deriv2', n);
B = A'; % Using matched case for clarity, but inequivalence holds generally
 
rng(0);
noise = randn(size(b_exact));
b_noise = b_exact + 1e-2 * norm(b_exact) * noise / norm(noise);
 
maxit = n;
tol = 1e-10;
lambda = 1e-3;

%% 2) Run PTR and RTP Solver Pairs
fprintf('Running PTR vs. RTP solver pairs...\n');

% BA Pair
[~, err_ba_ptr, ~, it_ba_ptr] = BAgmres_hybrid_bounds(A, B, b_noise, x_true, tol, maxit, lambda, zeros(size(B*A)));
[~, err_ba_rtp, ~, it_ba_rtp] = hybrid_ba_gmres_rtp(A, B, b_noise, x_true, tol, maxit, lambda);

% AB Pair
[~, err_ab_ptr, ~, it_ab_ptr] = ABgmres_hybrid_bounds(A, B, b_noise, x_true, tol, maxit, lambda, zeros(size(A*B)));
[~, err_ab_rtp, ~, it_ab_rtp] = hybrid_ab_gmres_rtp(A, B, b_noise, x_true, tol, maxit, lambda);

%% 3) Generate Plots
fprintf('Generating plots...\n');
figure('Name', 'PTR vs. RTP Inequivalence (Chapter 2)', 'Position', [200, 200, 1000, 500]);

% --- Plot 1: BA-GMRES (PTR vs RTP) ---
subplot(1, 2, 1);
semilogy(1:it_ba_ptr, err_ba_ptr, 'b-', 'LineWidth', 2, 'DisplayName', 'BA-GMRES (PTR)');
hold on;
semilogy(1:it_ba_rtp, err_ba_rtp, 'm-.', 'LineWidth', 2, 'DisplayName', 'BA-GMRES (RTP)');
hold off; grid on;
title('BA-GMRES: PTR vs. RTP (\neq)');
xlabel('Iteration k'); ylabel('Relative Error');
legend('show', 'Location', 'Best');

% --- Plot 2: AB-GMRES (PTR vs RTP) ---
subplot(1, 2, 2);
semilogy(1:it_ab_ptr, err_ab_ptr, 'b-', 'LineWidth', 2, 'DisplayName', 'AB-GMRES (PTR)');
hold on;
semilogy(1:it_ab_rtp, err_ab_rtp, 'm-.', 'LineWidth', 2, 'DisplayName', 'AB-GMRES (RTP)');
hold off; grid on;
title('AB-GMRES: PTR vs. RTP (\neq)');
xlabel('Iteration k');
legend('show', 'Location', 'Best');

sgtitle('Validation of PTR \neq RTP Inequivalence', 'FontSize', 16, 'FontWeight', 'bold');