function plot_error_surface()

clear all;
clc;

%% 1) Set up Test Problem and Parameters
n = 32;
problem_name = 'deriv2';
[A, b_exact, x_true] = generate_test_problem(problem_name, n);

rng(0); 
noise_level = 1e-2;
noise = randn(size(b_exact));
noise = noise / norm(noise) * noise_level * norm(b_exact);
b_noise = b_exact + noise;

E = 1e-4 * randn(size(A'));
B_pert = A' + E;
DeltaM_AB = A * E;
DeltaM_BA = E * A;
 
k_range = 1:n;
lambda_range = logspace(-8, -1, 100);
 
error_surface_ab = zeros(length(lambda_range), length(k_range));
error_surface_ba = zeros(length(lambda_range), length(k_range));

%% 2) Loop Through Parameters and Compute Errors for Both Methods 
tic; 
for i = 1:length(lambda_range)
    lambda = lambda_range(i);
    
    % Hybrid AB-GMRES 
    [~, err_hist_ab, ~, ~] = ABgmres_hybrid_bounds(A, B_pert, b_noise, x_true, 1e-10, n, lambda, DeltaM_AB);
    if length(err_hist_ab) < n, err_hist_ab(end+1:n) = NaN; end
    error_surface_ab(i, :) = err_hist_ab;
    
    % Hybrid BA-GMRES 
    [~, err_hist_ba, ~, ~] = BAgmres_hybrid_bounds(A, B_pert, b_noise, x_true, 1e-10, n, lambda, DeltaM_BA);
    if length(err_hist_ba) < n, err_hist_ba(end+1:n) = NaN; end
    error_surface_ba(i, :) = err_hist_ba;
    
    if mod(i, 20) == 0
        fprintf('   - Completed %d/%d lambda values...\n', i, length(lambda_range));
    end
end
toc;  

%% 3) Generate Plot for Hybrid AB-GMRES 

% optimal point for AB-GMRES
[min_val_ab, min_idx_flat_ab] = min(error_surface_ab(:));
[min_row_ab, min_col_ab] = ind2sub(size(error_surface_ab), min_idx_flat_ab);
optimal_k_ab = k_range(min_col_ab);
optimal_lambda_ab = lambda_range(min_row_ab);

fprintf(' Optimal Point (AB): k=%d, lambda=%.2e, Error=%.4f\n', optimal_k_ab, optimal_lambda_ab, min_val_ab);

figure('Name', 'Error Surface (Hybrid AB-GMRES)', 'Position', [100 100 800 650]);
imagesc(k_range, lambda_range, log10(error_surface_ab));
hold on;
plot(optimal_k_ab, optimal_lambda_ab, 'r*', 'MarkerSize', 15, 'LineWidth', 2, 'DisplayName', 'Optimal Point');
hold off;


set(gca, 'YDir', 'normal', 'YScale', 'log');
colorbarHandle = colorbar;
ylabel(colorbarHandle, 'log_{10}(Relative Error)');
xlabel('Iteration Count (k)');
ylabel('Regularization Parameter (\lambda)');
title('Error Surface for Hybrid AB-GMRES');
legend('show', 'Location', 'NorthEast');
grid on;
set(gca, 'FontSize', 12);

%% 4) Hybrid BA-GMRES
%  optimal point for BA-GMRES
[min_val_ba, min_idx_flat_ba] = min(error_surface_ba(:));
[min_row_ba, min_col_ba] = ind2sub(size(error_surface_ba), min_idx_flat_ba);
optimal_k_ba = k_range(min_col_ba);
optimal_lambda_ba = lambda_range(min_row_ba);

fprintf(' Optimal Point (BA): k=%d, lambda=%.2e, Error=%.4f\n', optimal_k_ba, optimal_lambda_ba, min_val_ba);

figure('Name', 'Error Surface (Hybrid BA-GMRES)', 'Position', [950 100 800 650]);
imagesc(k_range, lambda_range, log10(error_surface_ba));
hold on;
plot(optimal_k_ba, optimal_lambda_ba, 'r*', 'MarkerSize', 15, 'LineWidth', 2, 'DisplayName', 'Optimal Point');
hold off;

set(gca, 'YDir', 'normal', 'YScale', 'log');
colorbarHandle = colorbar;
ylabel(colorbarHandle, 'log_{10}(Relative Error)');
xlabel('Iteration Count (k)');
ylabel('Regularization Parameter (\lambda)');
title('Error Surface for Hybrid BA-GMRES');
legend('show', 'Location', 'NorthEast');
grid on;
set(gca, 'FontSize', 12);

fprintf('--- Analysis complete. ---\n');

end
