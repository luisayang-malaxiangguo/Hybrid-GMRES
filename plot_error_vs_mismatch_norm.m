function plot_error_vs_mismatch_norm() 

clear all; clc; close all;


n = 32;
problem_name = 'heat';
[A, b_exact, x_true] = generate_test_problem(problem_name, n);

rng(0); 
noise_level = 1e-2;
noise = randn(size(b_exact));
noise = noise / norm(noise) * noise_level * norm(b_exact);
b_noise = b_exact + noise;
 
E = randn(size(A'));
E = E / norm(E, 'fro'); 
maxit = n;
tol = 1e-6;

%%   

c_range = logspace(-8, -1, 20); 
mismatch_norms = zeros(size(c_range));
final_errors_ab = zeros(size(c_range));
final_errors_ba = zeros(size(c_range));

%% 
tic;
for i = 1:length(c_range)
    c = c_range(i);
     
    current_perturbation = c * E;
    B_pert = A' + current_perturbation;
     
    mismatch_norms(i) = norm(current_perturbation, 'fro');
     
    DeltaM_AB = A * current_perturbation;
    DeltaM_BA = current_perturbation * A;
    
    %  Find optimal lambda for each method at this perturbation level 
    k_gcv = 20;
    m = size(A,1);
    options = optimset('Display', 'off', 'TolX', 1e-8);

    gcv_handle_ab = @(lambda) gcv_function(lambda, A, B_pert, b_noise, m, k_gcv, 'ab');
    lambda_gcv_ab = fminbnd(gcv_handle_ab, 1e-9, 1e-1, options);
    
    gcv_handle_ba = @(lambda) gcv_function(lambda, A, B_pert, b_noise, m, k_gcv, 'ba');
    lambda_gcv_ba = fminbnd(gcv_handle_ba, 1e-9, 1e-1, options);
    
    %  Solve with each hybrid method and store the final error 
    [~, err_hist_ab, ~, ~] = ABgmres_hybrid_bounds(A, B_pert, b_noise, x_true, tol, maxit, lambda_gcv_ab, DeltaM_AB);
    final_errors_ab(i) = err_hist_ab(end);
    
    [~, err_hist_ba, ~, ~] = BAgmres_hybrid_bounds(A, B_pert, b_noise, x_true, tol, maxit, lambda_gcv_ba, DeltaM_BA);
    final_errors_ba(i) = err_hist_ba(end);
    
    fprintf('   - Level %d/%d complete. Mismatch Norm: %.2e, Error (AB): %.3f, Error (BA): %.3f\n', ...
            i, length(c_range), mismatch_norms(i), final_errors_ab(i), final_errors_ba(i));
end
toc;

%% 4) Plot
figure('Name', 'Error vs. Mismatch Norm', 'Position', [300 300 800 600]);
loglog(mismatch_norms, final_errors_ab, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Hybrid AB-GMRES');
hold on;
loglog(mismatch_norms, final_errors_ba, 'r-x', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Hybrid BA-GMRES');
hold off;

grid on;
title('Final Solution Error vs. Back-Projector Mismatch Norm');
xlabel('Mismatch Norm ||B - A^T||_F');
ylabel('Final Relative Error ||x_k - x_{true}|| / ||x_{true}||');
legend('show', 'Location', 'NorthWest');
axis tight;
set(gca, 'FontSize', 12);

end



