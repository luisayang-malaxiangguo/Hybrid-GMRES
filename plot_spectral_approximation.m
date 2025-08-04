function plot_spectral_approximation()
    %% 1) Set up Test Problem & Parameters
    n = 32;
    %[A, b, ~] = deriv2(n); 
    %[A, b, ~] = heat(n); 
    [A, b, ~] = shaw(n); 
    
    % Create a B that is NOT the transpose of A by adding a small perturbation
    B       = A';     % Matched back-projector
    % Use a fixed seed for the random perturbation for reproducibility
    rng(0);
    % Define perturbation E and perturbed matrix B
    E = 1e-5 * randn(size(A'));
    B_pert = A' + E;

    % Define the total perturbation terms for AB and BA methods
    DeltaM_AB = A * E;
    DeltaM_BA = E * A;

    % For the general case, we need the eigenvalues of the actual system operators.
    
    % Eigenvalues for the AB system
    M_ab = A * B;
    mu_ab_true = sort(real(eig(M_ab)), 'ascend');
    
    % Eigenvalues for the BA system
    M_ba = B * A;
    mu_ba_true = sort(real(eig(M_ba)), 'ascend');

    %% 2) Generate Plots
    k_values = [5, 15, 30]; 
    lambda = 1e-3; % Lambda for hybrid methods
    
    figure('Name', 'Figure 3: Spectral Approximation', 'Position', [100 100 850 700]);
    t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, 'Ritz / Harmonic Ritz Values (\theta) vs. True Eigenvalues (\mu)', 'FontSize', 14, 'FontWeight', 'bold');
    
    % --- Plot for non-hybrid AB ---
    ax1 = nexttile;
    % Pass the true eigenvalues of the AB operator
    plot_theta_vs_mu(ax1, 'ab_nonhybrid', A, B, b, mu_ab_true, k_values, lambda);
    title('non-hybrid AB-GMRES');
    ylabel('Value (log scale)');

    % --- Plot for non-hybrid BA ---
    ax2 = nexttile;
    % Pass the true eigenvalues of the BA operator
    plot_theta_vs_mu(ax2, 'ba_nonhybrid', A, B, b, mu_ba_true, k_values, lambda);
    title('non-hybrid BA-GMRES');
    
    % --- Plot for hybrid AB ---
    ax3 = nexttile;
    % Pass the true eigenvalues of the AB operator
    plot_theta_vs_mu(ax3, 'ab_hybrid', A, B, b, mu_ab_true, k_values, lambda);
    title('hybrid AB-GMRES');
    xlabel('Index'); ylabel('Value (log scale)');

    % --- Plot for hybrid BA ---
    ax4 = nexttile;
    % Pass the true eigenvalues of the BA operator
    plot_theta_vs_mu(ax4, 'ba_hybrid', A, B, b, mu_ba_true, k_values, lambda);
    title('hybrid BA-GMRES');
    xlabel('Index');
end

function plot_theta_vs_mu(ax, method, A, B, b, mu_true, k_vals, lambda)
    hold(ax, 'on');
    colors = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250];
    
    semilogy(ax, 1:length(mu_true), mu_true, 'k.', 'MarkerSize', 8, 'HandleVisibility', 'off');
    semilogy(ax, NaN, NaN, 'k-', 'LineWidth', 1.5, 'DisplayName', 'True \mu_i');
    
    for i = 1:length(k_vals)
        k = k_vals(i);
        Theta_k = get_spectral_values_stable(method, A, B, b, k, lambda);
        
        % FIX: Use the ACTUAL length of Theta_k for the x-coordinates
        % This handles cases where the Arnoldi loop breaks early.
        semilogy(ax, 1:length(Theta_k), Theta_k, 'o', 'MarkerSize', 6, 'Color', colors(i,:), ...
                 'MarkerFaceColor', colors(i,:), 'DisplayName', sprintf('k = %d', k));
    end
    
    hold(ax, 'off');
    grid on;
    legend('Location', 'SouthEast');
    xlim([0, length(mu_true) + 1]);
end
function Theta = get_spectral_values_stable(method, A, B, b, k_target, lambda)
    % This version correctly handles early breakdown AND sorts the output.
    maxit = k_target;
    if contains(method, 'ab')
        op = @(v) A * (B * v); r0 = b; op_size = size(A,1);
    else 
        op = @(v) B * (A * v); r0 = B*b; op_size = size(A,2);
    end

    Q = zeros(op_size, maxit + 1); 
    H = zeros(maxit + 1, maxit);
    beta = norm(r0); 
    Q(:,1) = r0 / beta;

    % --- Arnoldi Loop ---
    k = 0; % Initialize k outside the loop
    for k = 1:k_target
        v = op(Q(:,k));
        for j = 1:k
            H(j,k) = Q(:,j)'*v; 
            v = v - H(j,k)*Q(:,j);
        end
        H(k+1,k) = norm(v);
        if H(k+1,k) < 1e-12, break; end % Break if breakdown occurs
        Q(:,k+1) = v / H(k+1,k);
    end
    
    % --- Spectral Calculation ---
    % Use the FINAL value of k from the loop for slicing all matrices
    Hk_small = H(1:k, 1:k);

    if strcmp(method, 'ab_nonhybrid')
        [~, D_eig] = eig(Hk_small);
        Theta = real(diag(D_eig));
    else
        ek = zeros(k,1); 
        ek(end) = 1;
        
        y = pinv(Hk_small') * ek;
        P_unreg = Hk_small + (H(k+1, k)^2) * (y * ek');
        
        if contains(method, 'hybrid')
            P_reg = P_unreg + lambda*eye(k);
            [~, D_eig] = eig(P_reg);
            Theta = real(diag(D_eig));
        else
            [~, D_eig] = eig(P_unreg);
            Theta = real(diag(D_eig));
        end
    end
    
    % FIX: Sort Theta to match the sorting of mu_true for correct plotting
    Theta = sort(Theta, 'ascend');
end
