function [x, error_norm, residual_norm, niters, phi, dPhi] = ABgmres_hybrid_bounds_patched( ...
    A, B, b, x_true, tol, maxit, lambda, DeltaM)
% ABGMRES_HYBRID_BOUNDS_PATCHED AB-GMRES with Tikhonov + fully corrected perturbation bounds.
% PATCH NOTES:
%   1. Perturbation dK is now correctly formed as `A * DeltaM`.
%   2. Harmonic-Ritz values `Theta` are now computed using the explicit pencil
%      `P = Hk + h^2*Hk'\(ek*ek')`, then shifted by lambda, as specified.
%   3. Perturbations `dMu` are now computed in the same basis `W` as `dTheta`.

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
        if residual_norm(k) <= tol, break; end
    end
    niters        = k;
    x             = xk;
    residual_norm = residual_norm(1:k);
    error_norm    = error_norm(1:k);

    %--- SVD of A to get singular values sigmaA_i ---%
    [~, SA, ~] = svds(A, niters);
    sigmaA     = diag(SA);
    mu         = sigmaA.^2;

    %--- PATCHED: Form projected perturbation ---%
    Qk       = Q(:,1:k);
    dK       = A * DeltaM;
    dK_small = Qk' * dK * Qk;

    %--- PATCHED: Compute harmonic-Ritz values of M+lambda*I ---%
    Hk_small = H(1:k, 1:k);
    ek       = zeros(k,1);
    ek(end)  = 1;

    % Build the harmonic-Ritz pencil P for M and shift by lambda
    P = Hk_small + (H(k+1,k)^2) * (Hk_small'\(ek*ek'));
    P = P + lambda*eye(k);
    
    [W, D]     = eig(P);
    Theta      = real(diag(D));
    [Theta, p] = sort(Theta); % Ensure ascending order
    W          = W(:,p);      % Reorder eigenvectors accordingly

    %--- PATCHED: Compute perturbations dTheta and dMu using the same basis W ---%
    dTheta = real(diag(W' * dK_small * W));
    dMu    = dTheta; % For AB-GMRES, the shifts are computed in the same basis.

    %--- Build filter phi and bound dPhi ---%
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
    
    % First-order perturbation bound using formula from Theorem 4.7.1
    term1 = - (mu) .* sum((dTheta.'   ./ Theta.'.^2) .* P_excl, 2);
    term2 =   (lambda ./ s2l.^2)    .* (1 - P)        .* dMu;
    term3 =   (mu ./ s2l)       .* sum((1./Theta') .* P_excl, 2) .* dMu;
    dPhi  = term1 + term2 + term3;
end