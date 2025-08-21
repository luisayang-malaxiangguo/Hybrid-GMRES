function tomo_relevant_fig()
clear all; close all; clc;
addpath(genpath(pwd)); % Add all subfolders to the path
drawnow;
%% 1. Generate the Mismatched Tomography Test Problem
fprintf('1. Generating the tomography test problem (n=32)...\n'); drawnow;
n = 16; % A good compromise between speed and showing semi-convergence
options.CTtype = 'fancurved';
options.phantomImage = 'shepplogan';
[A_struct, b_exact, x_true] = PRtomo_mismatched(n, options);
A = A_struct.A; B = A_struct.B;
% Add a standard level of noise
rng(0); noise_level = 1e-2;
noise = randn(size(b_exact));
noise = noise / norm(noise) * noise_level * norm(b_exact);
b_noise = b_exact + noise;
% Define the perturbation matrices
E = B - A';
DeltaM_AB = A * E;
DeltaM_BA = E * A;
fprintf('   Problem generated successfully.\n\n'); drawnow;
%% 2. Set Solver Parameters & Find Optimal Lambdas
maxit = 100;
tol = 1e-8;
options_fmin = optimset('Display', 'off');
fprintf('2. Finding optimal lambdas via GCV for hybrid methods...\n'); drawnow;
% Find optimal lambda for AB-GMRES
gcv_handle_ab = @(l) gcv_function(l, A, B, b_noise, size(A,1), n, 'ab');
lambda_optimal_ab = fminbnd(gcv_handle_ab, 1e-6, 1e-2, options_fmin);
fprintf('   Optimal lambda for AB found: %.2e\n', lambda_optimal_ab); drawnow;
% Find optimal lambda for BA-GMRES
gcv_handle_ba = @(l) gcv_function(l, A, B, b_noise, size(A,1), n, 'ba');
lambda_optimal_ba = fminbnd(gcv_handle_ba, 1e-6, 1e-2, options_fmin);
fprintf('   Optimal lambda for BA found: %.2e\n\n', lambda_optimal_ba); drawnow;
%% PART 1: The Core Result (Performance & Reconstructions)
fprintf('--- PART 1: CORE PERFORMANCE ANALYSIS ---\n'); drawnow;
% --- Run Solvers Once with Optimal Lambdas ---
fprintf('   Running all four GMRES variants for core results...\n'); drawnow;
[x_ab_nh, err_ab_nh, res_ab_nh, it_ab_nh] = ABgmres_nonhybrid_bounds_TOMO(A, B, b_noise, x_true, 1e-14, maxit, DeltaM_AB);
[x_ba_nh, err_ba_nh, res_ba_nh, it_ba_nh] = BAgmres_nonhybrid_bounds_TOMO(A, B, b_noise, x_true, 1e-14, maxit, DeltaM_BA);
[x_ab_h, err_ab_h, res_ab_h, it_ab_h] = ABgmres_hybrid_bounds_TOMO(A, B, b_noise, x_true, tol, maxit, lambda_optimal_ab, DeltaM_AB);
[x_ba_h, err_ba_h, res_ba_h, it_ba_h] = BAgmres_hybrid_bounds_TOMO(A, B, b_noise, x_true, tol, maxit, lambda_optimal_ba, DeltaM_BA);
fprintf('   Solvers complete.\n'); drawnow;
% --- CONVERGENCE (ERROR & RESIDUAL) ---
figure('Name', 'Convergence');
subplot(1,2,1); semilogy(1:it_ab_nh, err_ab_nh, ':', 'DisplayName', 'non-hybrid AB'); hold on; semilogy(1:it_ba_nh, err_ba_nh, ':', 'DisplayName', 'non-hybrid BA'); semilogy(1:it_ab_h, err_ab_h, '-', 'DisplayName', 'hybrid AB'); semilogy(1:it_ba_h, err_ba_h, '-', 'DisplayName', 'hybrid BA'); hold off; title('Relative Error vs. Iteration'); xlabel('Iteration k'); grid on; legend('Location','Best');
subplot(1,2,2); semilogy(1:it_ab_nh, res_ab_nh, ':', 'DisplayName', 'non-hybrid AB'); hold on; semilogy(1:it_ba_nh, res_ba_nh, ':', 'DisplayName', 'non-hybrid BA'); semilogy(1:it_ab_h, res_ab_h, '-', 'DisplayName', 'hybrid AB'); semilogy(1:it_ba_h, res_ba_h, '-', 'DisplayName', 'hybrid BA'); hold off; title('Relative Residual vs. Iteration'); xlabel('Iteration k'); grid on; legend('Location','Best');
sgtitle('Convergence History');
% --- VISUAL COMPARISON OF RECONSTRUCTED SOLUTIONS (2D IMAGES) ---
figure('Name', 'Final Solutions (2D Images)');
clim_range = [min(x_true), max(x_true)];
subplot(2,2,1); imagesc(reshape(x_ab_nh, n, n)); colormap gray; axis image; axis off; title('non-hybrid AB'); caxis(clim_range);
subplot(2,2,2); imagesc(reshape(x_ba_nh, n, n)); colormap gray; axis image; axis off; title('non-hybrid BA'); caxis(clim_range);
subplot(2,2,3); imagesc(reshape(x_ab_h, n, n)); colormap gray; axis image; axis off; title(sprintf('hybrid AB (\\lambda=%.1e)', lambda_optimal_ab)); caxis(clim_range);
subplot(2,2,4); imagesc(reshape(x_ba_h, n, n)); colormap gray; axis image; axis off; title(sprintf('hybrid BA (\\lambda=%.1e)', lambda_optimal_ba)); caxis(clim_range);
sgtitle('Comparison of Final Reconstructed Solutions');

% --- VISUAL COMPARISON OF RECONSTRUCTED SOLUTIONS (1D VECTORS) ---
figure('Name', 'Final Solutions (1D Vectors)');
plot(x_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True Solution');
hold on;
plot(x_ab_nh, ':', 'DisplayName', 'non-hybrid AB (\lambda=0)');
plot(x_ba_nh, ':', 'DisplayName', 'non-hybrid BA (\lambda=0)');
plot(x_ab_h, '-', 'DisplayName', sprintf('hybrid AB (\\lambda=%.1e)', lambda_optimal_ab));
plot(x_ba_h, '-', 'DisplayName', sprintf('hybrid BA (\\lambda=%.1e)', lambda_optimal_ba));
hold off;
title('Comparison of Final Solutions (1D Vector View)');
xlabel('Element Index');
ylabel('Value');
legend('Location', 'Best');
grid on;
xlim([1, length(x_true)]);

fprintf('   Part 1 plots generated.\n\n'); drawnow;
%% PART 2: The Justification (Parameter Selection)
fprintf('--- PART 2: REGULARIZATION PARAMETER ANALYSIS ---\n'); drawnow;
% --- L-CURVE AND ERROR VS LAMBDA ---
fprintf('   Running L-curve analysis...\n'); drawnow;
lambda_range = logspace(-7, -2, 40);
res_ab=zeros(size(lambda_range)); sol_ab=zeros(size(lambda_range)); err_ab=zeros(size(lambda_range));
res_ba=zeros(size(lambda_range)); sol_ba=zeros(size(lambda_range)); err_ba=zeros(size(lambda_range));
for i = 1:length(lambda_range)
    lambda_i = lambda_range(i);
    [x_ab_i, err_ab_i] = ABgmres_hybrid_bounds_TOMO(A, B, b_noise, x_true, tol, maxit, lambda_i, DeltaM_AB);
    res_ab(i) = norm(b_noise - A * x_ab_i); sol_ab(i) = norm(x_ab_i); err_ab(i) = err_ab_i(end);
    [x_ba_i, err_ba_i] = BAgmres_hybrid_bounds_TOMO(A, B, b_noise, x_true, tol, maxit, lambda_i, DeltaM_BA);
    res_ba(i) = norm(b_noise - A * x_ba_i); sol_ba(i) = norm(x_ba_i); err_ba(i) = err_ba_i(end);
end
[~, idx_opt_ab] = min(abs(lambda_range - lambda_optimal_ab));
[~, idx_opt_ba] = min(abs(lambda_range - lambda_optimal_ba));
figure('Name', 'L-Curve and Error vs Lambda');
subplot(2,2,1); loglog(res_ab, sol_ab, 'b-o','DisplayName','L-Curve'); hold on; plot(res_ab(idx_opt_ab), sol_ab(idx_opt_ab), 'r*','DisplayName','Optimal \lambda_{AB}'); title('L-Curve (AB)'); xlabel('Residual Norm'); ylabel('Solution Norm'); grid on; legend;
subplot(2,2,2); loglog(lambda_range, err_ab, 'b-o','DisplayName','Error'); hold on; plot(lambda_optimal_ab, err_ab(idx_opt_ab), 'r*','DisplayName','Optimal \lambda_{AB}'); title('Error vs. Lambda (AB)'); xlabel('Lambda'); ylabel('Relative Error'); grid on; legend;
subplot(2,2,3); loglog(res_ba, sol_ba, 'm-x','DisplayName','L-Curve'); hold on; plot(res_ba(idx_opt_ba), sol_ba(idx_opt_ba), 'r*','DisplayName','Optimal \lambda_{BA}'); title('L-Curve (BA)'); xlabel('Residual Norm'); ylabel('Solution Norm'); grid on; legend;
subplot(2,2,4); loglog(lambda_range, err_ba, 'm-x','DisplayName','Error'); hold on; plot(lambda_optimal_ba, err_ba(idx_opt_ba), 'r*','DisplayName','Optimal \lambda_{BA}'); title('Error vs. Lambda (BA)'); xlabel('Lambda'); ylabel('Relative Error'); grid on; legend;
sgtitle('Parameter Selection Analysis');
fprintf('   Part 2 plots generated.\n\n'); drawnow;
%% PART 3: The Explanation (Theoretical Validation)
fprintf('--- PART 3: THEORETICAL VALIDATION ---\n'); drawnow;
% --- FILTER FACTORS ---
fprintf('   Generating filter factor plots...\n'); drawnow;
[U,S,V] = svd(full(A),'econ'); sigma = diag(S);
d_svd = U' * b_noise; d_svd(abs(d_svd) < 1e-12) = 1;
[x_ab_nh_ff, ~, ~, ~, phi_ab_nh_ff] = ABgmres_nonhybrid_bounds_TOMO(A, B, b_noise, x_true, tol, n, DeltaM_AB);
[x_ba_nh_ff, ~, ~, ~, phi_ba_nh_ff] = BAgmres_nonhybrid_bounds_TOMO(A, B, b_noise, x_true, tol, n, DeltaM_BA);
[x_ab_h_ff, ~, ~, ~, phi_ab_h_ff] = ABgmres_hybrid_bounds_TOMO(A, B, b_noise, x_true, tol, n, lambda_optimal_ab, DeltaM_AB);
[x_ba_h_ff, ~, ~, ~, phi_ba_h_ff] = BAgmres_hybrid_bounds_TOMO(A, B, b_noise, x_true, tol, n, lambda_optimal_ba, DeltaM_BA);
phi_emp_ab_nh = sigma .* (V' * x_ab_nh_ff) ./ d_svd; phi_emp_ba_nh = sigma .* (V' * x_ba_nh_ff) ./ d_svd;
phi_emp_ab_h = sigma .* (V' * x_ab_h_ff) ./ d_svd; phi_emp_ba_h = sigma .* (V' * x_ba_h_ff) ./ d_svd;
figure('Name', 'Final Filter Factor Comparison');
subplot(2,2,1); plot(real(phi_ab_nh_ff), '--'); hold on; plot(real(phi_emp_ab_nh(1:length(phi_ab_nh_ff))), 'o-'); title('non-hybrid AB'); legend('Theoretical','Empirical');
subplot(2,2,2); plot(real(phi_ba_nh_ff), '--'); hold on; plot(real(phi_emp_ba_nh(1:length(phi_ba_nh_ff))), 'o-'); title('non-hybrid BA');
subplot(2,2,3); plot(real(phi_ab_h_ff), '--'); hold on; plot(real(phi_emp_ab_h(1:length(phi_ab_h_ff))), 'o-'); title('hybrid AB');
subplot(2,2,4); plot(real(phi_ba_h_ff), '--'); hold on; plot(real(phi_emp_ba_h(1:length(phi_ba_h_ff))), 'o-'); title('hybrid BA');
sgtitle('Theoretical vs. Empirical Filter Factors');
fprintf('   Part 3 plots generated.\n\n'); drawnow;
fprintf('\n--- All Relevant Analyses Complete ---\n');
end