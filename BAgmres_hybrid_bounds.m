function [x, error_norm, residual_norm, niters, phi, dPhi] = BAgmres_hybrid_bounds_corrected( ...
    A, B, b, x_true, tol, maxit, lambda, DeltaM)
% BAGMRES_HYBRID_BOUNDS_CORRECTED BA-GMRES with Tikhonov + corrected perturbation bounds.
% CORRECTIONS:
%   1. 'Theta' now correctly computes the harmonic-Ritz values of the regularized
%      operator M+lambda*I by solving a generalized eigenvalue problem.
%   2. 'dTheta' uses the corresponding harmonic-Ritz vectors.
%   3. 'dMu' calculation now correctly uses VA (eigenvectors of M = A'A) instead of UA.

    % Build M = B*A and its perturbation matrix
    M  = B * A;
    dK = M' * DeltaM + DeltaM' * M;

    %--- Arnoldi + Tikhonov on M = B*A ---
    n     = size(A,2);
    x0    = zeros(n,1);
    r0    = B * (b - A*x0);
    beta  = norm(r0);
    Q     = zeros(n, maxit+1);
    H     = zeros(maxit+1, maxit);
    Q(:,1)= r0/beta;
    e1    = [beta; zeros(maxit,1)];
    residual_norm = zeros(maxit,1);
    error_norm    = zeros(maxit,1);
    for k = 1:maxit
        % Arnoldi step
        v = B*(A*Q(:,k));
        for j = 1:k
            H(j,k) = Q(:,j)'*v;
            v       = v - H(j,k)*Q(:,j);
        end
        H(k+1,k) = norm(v);
        if H(k+1,k)==0, break; end
        Q(:,k+1) = v/H(k+1,k);
        % Projected Tikhonov subproblem
        Hk = H(1:k+1,1:k);
        tk = e1(1:k+1);
        yk = (Hk'*Hk + lambda*eye(k)) \ (Hk'*tk);
        xk = Q(:,1:k)*yk;
        % Norms & stopping
        residual_norm(k) = norm(b - A*xk)/norm(b);
        error_norm(k)    = norm(xk - x_true)/norm(x_true);
        if residual_norm(k) <= tol, break; end
    end
    niters        = k;
    x             = xk;
    residual_norm = residual_norm(1:k);
    error_norm    = error_norm(1:k);

    %--- SVD of A to get singular values sigmaA_i ---%
    [UA, SA, VA] = svds(A, niters);
    sigmaA       = diag(SA);
    mu           = sigmaA.^2;

    %--- CORRECTED: Compute harmonic-Ritz values and their shifts dTheta ---%
    Qk          = Q(:,1:k);
    dK_small    = Qk' * (DeltaM * Qk); % Perturbation of projected M
    Hk_full     = H(1:k+1, 1:k);
    Hk_small    = H(1:k, 1:k);

    % Build Hessenberg matrices for the regularized operator K = M + lambda*I
    Hk_reg_full  = Hk_full  + lambda * [eye(k); zeros(1, k)];
    Hk_reg_small = Hk_small + lambda * eye(k);

   % Solve generalized eigenproblem for harmonic-Ritz values of K 
    [W, Th]  = eig(Hk_reg_full' * Hk_reg_full, Hk_reg_small);
    Theta    = real(diag(Th));
    [Theta, p] = sort(Theta); % ensure ascending order
    W        = W(:,p); % reorder harmonic-Ritz vectors

    % Perturbation of projected K is dK_small.
    dTheta   = real(diag(W' * dK_small * W));

    %--- CORRECTED: Compute singular-value shifts dMu for mu_i = (sigmaA_i)^2 ---%
    % The eigenvectors of M = A'A are the columns of VA. The original code incorrectly used UA.
    % The formula calculates the diagonal of VA' * DeltaM * VA.
    dMu = sum(VA .* (DeltaM * VA), 1)';

    %--- Build filter phi and bound dPhi (the logic here was mostly correct) ---%
    s2l   = mu + lambda;
    eps0  = eps;
    Clog  = zeros(k,1);
    for i = 1:k
        terms    = max(1 - s2l(i)./Theta.', eps0);
        Clog(i)  = sum(log(terms));
    end
    P = exp(Clog);
    
    P_excl = zeros(k,k);
    for i = 1:k
      for j = 1:k
        denom        = max(1 - s2l(i)/Theta(j), eps0);
        P_excl(i,j)  = exp(Clog(i) - log(denom));
      end
    end
    
    phi_z = (mu ./ s2l) .* (1 - P);
    phi   = sigmaA .* phi_z; % This scaling is not in the PDF but kept from original code
    
    term1 = - (mu) .* sum((dTheta.'   ./ Theta.'.^2) .* P_excl, 2);
    term2 =   (lambda ./ s2l.^2)    .* (1 - P)        .* dMu;
    term3 =   (mu ./ s2l)       .* sum((1./Theta') .* P_excl, 2) .* dMu;
    dPhi  = term1 + term2 + term3;
end