function plot_spectral_analysis()
% PLOT_SPECTRAL_ANALYSIS Visualizes the convergence of Ritz and harmonic
% Ritz values to the true eigenvalues for all four GMRES variants.
%
% This script generates a 2x2 plot where each subplot corresponds to one
% GMRES method. It shows the true eigenvalues of the system matrix and

% overlays the spectral approximations computed at different iterations.

clear all;
clc;
close all;

%% 1) Set up Test Problem and True Eigenvalues
fprintf('1. Setting up the test problem and computing true eigenvalues...\n');
n = 32;
problem_name = 'shaw';
[A, b_exact, ~] = generate_test_problem(problem_name, n);

% For this analysis, we use a slightly perturbed back-projector
rng(0);
E = 1e-4 * randn(size(A'));
B_pert = A' + E;

% --- Compute and sort the true eigenvalues for both system matrices ---
M_ab = A * B_pert;
mu_ab_true = sort(real(eig(M_ab)), 'ascend');

M_ba = B_pert * A;
mu_ba_true = sort(real(eig(M_ba)), 'ascend');

%% 2) Generate the 2x2 Plot
fprintf('2. Generating spectral approximation plots...\n');
k_values = [5, 15, 30]; % Iterations to visualize
lambda = 1e-3; % A representative lambda for hybrid methods

figure('Name', 'Spectral Approximation Analysis', 'Position', [100 100 900 750]);
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, 'Ritz / Harmonic Ritz Values (\theta) vs. True Eigenvalues (\mu)', 'FontSize', 14, 'FontWeight', 'bold');

% --- Subplot for non-hybrid AB-GMRES (uses Ritz Values) ---
ax1 = nexttile;
plot_single_method_spectrum(ax1, 'nonhybrid_ab', A, B_pert, b_exact, mu_ab_true, k_values, lambda);
title('non-hybrid AB-GMRES');
ylabel('Value (log scale)');

% --- Subplot for non-hybrid BA-GMRES (uses Harmonic Ritz Values) ---
ax2 = nexttile;
plot_single_method_spectrum(ax2, 'nonhybrid_ba', A, B_pert, b_exact, mu_ba_true, k_values, lambda);
title('non-hybrid BA-GMRES');

% --- Subplot for hybrid AB-GMRES (uses Harmonic Ritz of regularized op) ---
ax3 = nexttile;
plot_single_method_spectrum(ax3, 'hybrid_ab', A, B_pert, b_exact, mu_ab_true, k_values, lambda);
title('hybrid AB-GMRES');
xlabel('Eigenvalue Index');
ylabel('Value (log scale)');

% --- Subplot for hybrid BA-GMRES (uses Harmonic Ritz of regularized op) ---
ax4 = nexttile;
plot_single_method_spectrum(ax4, 'hybrid_ba', A, B_pert, b_exact, mu_ba_true, k_values, lambda);
title('hybrid BA-GMRES');
xlabel('Eigenvalue Index');

fprintf('--- Analysis complete. ---\n');
end

%% Helper function to plot the spectrum for a single method
function plot_single_method_spectrum(ax, method_type, A, B, b, mu_true, k_vals, lambda)
    hold(ax, 'on');
    colors = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250];
    
    % Plot true eigenvalues as black dots
    semilogy(ax, 1:length(mu_true), mu_true, 'k.', 'MarkerSize', 12, 'DisplayName', 'True \mu_i');
    
    % For each k, compute and plot the spectral approximations
    for i = 1:length(k_vals)
        k = k_vals(i);
        theta_k = get_spectral_approximations(method_type, A, B, b, k, lambda);
        
        % Plot the computed thetas
        if ~isempty(theta_k)
            semilogy(ax, 1:length(theta_k), theta_k, 'o', 'MarkerSize', 7, ...
                     'Color', colors(i,:), 'MarkerFaceColor', colors(i,:), ...
                     'DisplayName', sprintf('\\theta_j for k = %d', k));
        end
    end
    
    hold(ax, 'off');
    grid on;
    legend('Location', 'SouthEast');
    xlim([0, length(mu_true) + 1]);
    set(gca, 'FontSize', 11);
end

%% Helper function to run Arnoldi and compute spectral approximations
function theta = get_spectral_approximations(method_type, A, B, b, k_target, lambda)
    % --- Set up the operator and starting vector based on method type ---
    if contains(method_type, 'ab')
        op = @(v) A * (B * v);
        r0 = b;
        op_size = size(A, 1);
    else % 'ba'
        op = @(v) B * (A * v);
        r0 = B * b;
        op_size = size(A, 2);
    end

    % --- Arnoldi Process ---
    Q = zeros(op_size, k_target + 1);
    H = zeros(k_target + 1, k_target);
    beta = norm(r0);
    if beta == 0, theta = []; return; end
    Q(:,1) = r0 / beta;

    k_actual = 0; % Actual number of iterations before breakdown
    for k = 1:k_target
        v = op(Q(:,k));
        for j = 1:k
            H(j,k) = Q(:,j)' * v;
            v = v - H(j,k) * Q(:,j);
        end
        H(k+1,k) = norm(v);
        if H(k+1,k) < 1e-12
            k_actual = k;
            break; % Breakdown
        end
        Q(:,k+1) = v / H(k+1,k);
        k_actual = k;
    end
    
    % --- Compute Spectral Approximations from Hessenberg matrix H ---
    Hk_small = H(1:k_actual, 1:k_actual);

    switch method_type
        case 'nonhybrid_ab'
            % Non-hybrid AB is FOM-like, uses Ritz values of M
            theta = real(eig(Hk_small));
            
        case 'nonhybrid_ba'
            % Non-hybrid BA is GMRES-like, uses harmonic Ritz values of M
            ek = zeros(k_actual, 1); ek(end) = 1;
            % The harmonic Ritz values are the eigenvalues of this pencil:
            P_matrix = Hk_small + (H(k_actual+1, k_actual)^2) * (Hk_small' \ (ek * ek'));
            theta = real(eig(P_matrix));
            
        case {'hybrid_ab', 'hybrid_ba'}
            % Hybrid methods use harmonic Ritz values of the regularized operator
            ek = zeros(k_actual, 1); ek(end) = 1;
            P_unreg = Hk_small + (H(k_actual+1, k_actual)^2) * (Hk_small' \ (ek * ek'));
            P_reg = P_unreg + lambda * eye(k_actual);
            theta = real(eig(P_reg));
    end
    
    % Sort for consistent plotting
    theta = sort(theta, 'ascend');
end
