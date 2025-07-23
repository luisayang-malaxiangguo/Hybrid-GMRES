function [x, error_norm, residual_norm, niters, phi_final, dphi_final, phi_iter, dphi_iter] = ABgmres_hybrid_bounds( ...
    A, B, b, x_true, tol, maxit, lambda, DeltaM)
% ABGMRES_HYBRID_BOUNDS AB-GMRES with Tikhonov regularization and perturbation bounds.
% This function returns filter factors (phi) and their perturbation bounds (dPhi) at each iteration.

    %--- Arnoldi + Tikhonov ---
    m     = size(A,1);
    z0    = zeros(size(B,2),1);
    r0    = b - A*(B*z0);
    beta  = norm(r0);
    Q     = zeros(m, maxit+1);
    H     = zeros(maxit+1, maxit);
    Q(:,1)= r0 / beta;
    e1    = [beta; zeros(maxit,1)];
    residual_norm = zeros(maxit,1);
    error_norm    = zeros(maxit,1);
    
    % Initialize cell arrays
    phi_iter = cell(maxit, 1);
    dphi_iter = cell(maxit, 1);

    % --- EFFICIENT SVD: Calculate only ONCE before the loop ---
    [UA, SA, ~] = svds(A, maxit); % Need UA for this method
    sigmaA = diag(SA);
    mu_full = sigmaA.^2;

    for k = 1:maxit
        % Arnoldi step on M = A*B
        v = A*(B*Q(:,k));
        for j = 1:k
            H(j,k) = Q(:,j)'*v;
            v      = v - H(j,k)*Q(:,j);
        end
        H(k+1,k) = norm(v);
        if H(k+1,k) == 0, break; end
        Q(:,k+1) = v / H(k+1,k);
        
        % Projected Tikhonov solve
        Hk = H(1:k+1,1:k);
        tk = e1(1:k+1);
        yk = (Hk'*Hk + lambda*eye(k)) \ (Hk'*tk);
        zk = Q(:,1:k)*yk;
        xk = B * zk;
        
        % Norms & stopping
        residual_norm(k) = norm(b - A*xk)/norm(b);
        error_norm(k)    = norm(xk - x_true)/norm(x_true);
        
        % =====================================================================
        % --- THIS IS THE CORRECT CALCULATION BLOCK FOR HYBRID AB-GMRES ---
        % =====================================================================
        
        % Form projected perturbation
        Qk_current = Q(:,1:k);
        dK_current = A * DeltaM;
        dK_small_current = Qk_current' * dK_current * Qk_current;

        % Compute harmonic-Ritz values of M + lambda*I
        Hk_small_current = H(1:k, 1:k);
        ek_current       = zeros(k,1);
        ek_current(end)  = 1;
        % Harmonic-Ritz values of the unregularized operator M
        P_unreg = Hk_small_current + (H(k+1,k)^2) * (Hk_small_current'\(ek_current*ek_current'));
        % The harmonic-Ritz values of K = M + lambda*I are the eigenvalues of P_unreg + lambda*I
        P_reg = P_unreg + lambda*eye(k);
        [W_current, D_eig] = eig(P_reg);
        Theta_current = real(diag(D_eig));
        [Theta_current, p_sort] = sort(Theta_current); % Ensure ascending order
        W_current = W_current(:,p_sort);      % Reorder eigenvectors accordingly

        % Compute perturbations dTheta and dMu
        dTheta_current = real(diag(W_current' * dK_small_current * W_current));
        dMu_current    = dTheta_current; % For AB-GMRES, the shifts are computed in the same basis.

        % Truncate mu to the current size k
        mu_current = mu_full(1:k);

        % Build phi and dphi using filter factor formula from Theorem 4.5.1
        s2l = mu_current + lambda;
        eps0_current = eps;
        Clog_current = zeros(k,1);
        P_excl_current = zeros(k,k);
        for i = 1:k
            terms = max(1 - s2l(i)./Theta_current.', eps0_current);
            Clog_current(i) = sum(log(terms));
            for j = 1:k
                denom = max(1 - s2l(i)/Theta_current(j), eps0_current);
                P_excl_current(i,j) = exp(Clog_current(i) - log(denom));
            end
        end
        P_final = exp(Clog_current);
        
        % CORRECT formula for HYBRID filter factor
        phi_current = (mu_current ./ s2l) .* (1 - P_final);
        
        % CORRECT formula for HYBRID perturbation bound (3 terms)
        term1 = - (mu_current) .* sum((dTheta_current.' ./ Theta_current.'.^2) .* P_excl_current, 2);
        term2 =   (lambda ./ s2l.^2) .* (1 - P_final) .* dMu_current;
        term3 =   (mu_current ./ s2l) .* sum((1./Theta_current') .* P_excl_current, 2) .* dMu_current;
        dphi_current = term1 + term2 + term3;

        % Store the calculated phi and dphi in our cell arrays
        phi_iter{k} = phi_current;
        dphi_iter{k} = dphi_current;

        % Original stopping condition
        if residual_norm(k) <= tol, break; end
    end

    niters = k;
    x = xk;
    residual_norm = residual_norm(1:k);
    error_norm = error_norm(1:k);

    % The final values are from the last iteration
    phi_final = phi_iter{k};
    dphi_final = dphi_iter{k};

    % Trim the cell arrays to the actual number of iterations performed
    phi_iter = phi_iter(1:k);
    dphi_iter = dphi_iter(1:k);
end
