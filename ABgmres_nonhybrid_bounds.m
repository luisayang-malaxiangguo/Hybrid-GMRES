function [x, err, res, niters, phi, dPhi] = ABgmres_nonhybrid_bounds( ...
    A, B, b, x_true, tol, maxit, DeltaM)
% ABGMRES_NONHYBRID_BOUNDS_PATCHED Non-hybrid AB-GMRES + fully corrected perturbation bounds.
% PATCH NOTES:
%   1. `Theta` now correctly computes the harmonic-Ritz values of M using the
%      explicit pencil `P = Hk + h^2*Hk'\(ek*ek')`.
%   2. `dMu` is now computed in the same harmonic-Ritz basis `W` as `dTheta`.
%   3. The projected perturbation `dKk` is formed more clearly.

    % Build M = A*B
    M = A * B;
    
    %--- Arnoldi GMRES on M z = b ---%
    m     = size(A,1);
    z0    = zeros(size(B,2),1);
    r0    = b - A*(B*z0);
    beta  = norm(r0);
    Q     = zeros(m, maxit+1);
    H     = zeros(maxit+1, maxit);
    Q(:,1)= r0 / beta;
    res   = zeros(maxit,1);
    err   = zeros(maxit,1);
    for k = 1:maxit
        % Arnoldi step
        v = A*(B*Q(:,k));
        for j = 1:k
            H(j,k) = Q(:,j)' * v;
            v      = v - H(j,k) * Q(:,j);
        end
        H(k+1,k) = norm(v);
        if H(k+1,k)==0, break; end
        Q(:,k+1) = v / H(k+1,k);
        % Solve projected least-squares Hk*y = beta*e1
        Hk = H(1:k+1,1:k);
        yk = Hk \ ([beta; zeros(k,1)]);
        z  = Q(:,1:k) * yk;
        xk = B * z;
        % Compute norms & check convergence
        res(k) = norm(b - A*xk)/norm(b);
        err(k) = norm(xk - x_true)/norm(x_true);
        if res(k) <= tol, break; end
    end
    niters = k;
    x      = xk;
    res    = res(1:k);
    err    = err(1:k);

    % SVD of A for sigma_i and mu_i = sigma_i^2
    [~,S,~] = svd(A,'econ');
    mu      = diag(S).^2;

    %--- PATCHED: Compute harmonic-Ritz values and their perturbations ---%
    Qk       = Q(:,1:k);
    Hk_small = H(1:k,1:k);

    % For M=AB, the perturbation on M is dM = A*DeltaB
    dKk = Qk'*(A*DeltaM * Qk);

    % Compute harmonic-Ritz values of M using the explicit formula
    ek = zeros(k,1); 
    ek(end) = 1;
    P = Hk_small + (H(k+1,k)^2)*(Hk_small'\(ek*ek'));
    
    [W, Th]    = eig(P);
    Theta      = real(diag(Th));
    [Theta, p] = sort(Theta); % sort ascending
    W          = W(:, p);     % reorder harmonic-Ritz vectors
    
    % Compute perturbations for Theta and Mu in the same harmonic-Ritz basis W
    dTheta = real(diag(W' * dKk * W));
    dMu    = real(diag(W' * dKk * W));

    % Truncate to k modes
    mu  = mu(1:k);
    dMu = dMu(1:k);

    %--- Build phi and dphi using filter factor formula from Theorem 4.3.1 ---%
    eps0   = eps;
    Clog   = zeros(k,1);
    P_excl = zeros(k,k);
    for i = 1:k
        factors   = max(1 - mu(i)./Theta.', eps0);
        Clog(i)   = sum(log(factors));
        for j = 1:k
            denom        = max(1 - mu(i)/Theta(j), eps0);
            P_excl(i,j)  = exp(Clog(i) - log(denom));
        end
    end
    P   = exp(Clog);
    phi = 1 - P; % Filter factors
    
    % First-order perturbation bound
    term1 = - mu .* sum((dTheta' ./ Theta'.^2) .* P_excl, 2);
    term2 =   sum((1./Theta') .* P_excl, 2)    .* dMu;
    dPhi  = term1 + term2;
end