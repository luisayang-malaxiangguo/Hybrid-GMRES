​function [x, error_norm, residual_norm, niters, phi, dPhi] = BAgmres_hybrid_bounds(A, B, b, x_true, tol, maxit, lambda, DeltaM)

% BAGMRES_HYBRID_BOUNDS BA-GMRES with Tikhonov + first-order perturbation bounds
% Additional outputs:
% phi — filter factors at final iteration (size = n)
% dPhi — first-order bound on perturbation of phi (size = n)
% Inputs:
% A, B – forward operator and preconditioner (B*A is n×n)
% b, x_true – data and true solution
% tol – stopping tolerance on ||b–A*x||/||b||
% maxit – maximum GMRES iterations
% lambda – Tikhonov parameter
% DeltaM – n×n perturbation on M = B*A
%
% Outputs:
% x – computed solution
% error_norm – ||x_k–x_true||/||x_true|| at each iterate
% residual_norm – ||b–A*x_k||/||b|| at each iterate
% niters – number of iterations performed (k)
% phi, dPhi – length-k vectors of unperturbed filter factors
% and their first-order perturbation bounds
% Build M and its perturbation ΔK
    M = B * A; % n×n
    dK = M' * DeltaM + DeltaM' * M;

    %— Arnoldi + Tikhonov GMRES on BA —%
    n = size(A,2);
    x0 = zeros(n,1);
    r0 = B * (b - A*x0);
    beta = norm(r0);
    Q = zeros(n, maxit+1);
    H = zeros(maxit+1, maxit);
    Q(:,1) = r0 / beta;
    e1 = [beta; zeros(maxit,1)];
    residual_norm = zeros(maxit,1);
    error_norm = zeros(maxit,1);

    for k = 1:maxit
        % Arnoldi step on M = B*A
        v = B * (A * Q(:,k));
        for j = 1:k
            H(j,k) = Q(:,j)' * v;
            v = v - H(j,k) * Q(:,j);
        end
        H(k+1,k) = norm(v);
        if H(k+1,k) == 0, break; end
    Q(:,k+1) = v / H(k+1,k);

    % Tikhonov in the small subproblem
    Hk = H(1:k+1,1:k); tk = e1(1:k+1);
    yk = (Hk'*Hk + lambda*eye(k)) \ (Hk'*tk);
    xk = Q(:,1:k) * yk;

    % norms & stopping
    rk = b - A * xk;
    residual_norm(k) = norm(rk)/norm(b);
    error_norm(k) = norm(xk - x_true)/norm(x_true);
    if residual_norm(k) <= tol, break; end
    end

    niters = k;
    x = xk;
    residual_norm = residual_norm(1:k);
    error_norm = error_norm(1:k);

    %— Truncated SVD on M to get top-k singular triplets —%
    [~, S, V] = svds(M, niters);
    sigma = diag(S); % k×1

    %— Compute first-order Ritz-value shifts dTheta —%
    Qk = Q(:,1:k);
    Hk = H(1:k,1:k);
    dK_small = Qk' * (dK * Qk);
    [Vh, Th] = eig(Hk);
    Theta_raw = diag(Th); % may contain tiny values

    % Threshold Ritz values to avoid overflow
    theta_min = 1e-12;
    Theta = max(abs(Theta_raw), theta_min);
    dTheta = diag(Vh' * (dK_small * Vh)); % k×1

    %— Compute first-order singular-value shifts dMu —%
    MU = M * V;
    DMV = DeltaM * V;
    dMu = sum(V .* (M.'*DMV + DeltaM.'*MU), 1).'; % k×1

    %— Vectorized computation of φ and perturbations dPhi —%
    s2l = sigma.^2 + lambda; % k×1

    % Build log-domain product: log(C_i) = sum_j log(1 - s2l_i/Theta_j)
    % where Theta_j ≥ theta_min ensures (1 - ...) bounded away from -Inf
    Clog = zeros(k,1);
    for i = 1:k
        Clog(i) = sum(log( max(1 - s2l(i)./Theta.', -1+1e-16) ));
    end
    P = exp(Clog); % ∏_j (1 - s2l/Theta_j)

    % Recover P_excl via P_excl(i,j) = P(i)/(1 - s2l(i)/Theta(j))
    % but compute safely in log-domain:
    P_excl = zeros(k,k);
    for i = 1:k
        for j = 1:k
            term = 1 - s2l(i)/Theta(j);
            term = max(term, -1+1e-16);
            P_excl(i,j) = exp(Clog(i) - log(term));
        end
    end

    % φ (unperturbed filter factors)
    phi = (sigma.^2 ./ s2l) .* (1 - P);

    % dPhi (first-order perturbation bounds)
    term1 = - sigma.^2 .* sum((dTheta.' ./ Theta.'.^2) .* P_excl, 2);
    term2 = (lambda ./ s2l.^2) .* (1 - P) .* dMu;
    term3 = (sigma.^2 ./ s2l) .* sum((1./Theta') .* P_excl, 2) .* dMu;
    dPhi = term1 + term2 + term3;
end