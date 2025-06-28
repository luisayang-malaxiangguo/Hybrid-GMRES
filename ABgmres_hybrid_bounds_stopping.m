function [x, error_norm, residual_norm, niters, phi, dPhi, lambda_vec] = ABgmres_hybrid_bounds_stopping( ...
    A, B, b, x_true, tol, maxit, lambda, DeltaM, method)
% ABGMRES_HYBRID_BOUNDS_STOPPING  AB-GMRES with Tikhonov + DP/GCV-based λ + 1st-order bounds
%
%   [x, err, res, k, phi, dPhi, lambda_vec] = ...
%     ABgmres_hybrid_bounds_stopping(A,B,b,x_true,tol,maxit,lambda,DeltaM,method)
%
% Inputs:
%   A, B       – forward operator (m×n) and back-projector (n×m)
%   b, x_true  – data and true solution (for DP noise estimate)
%   tol        – stopping tolerance on ||b–A*x||/||b||
%   maxit      – maximum GMRES iterations
%   lambda     – fallback Tikhonov parameter
%   DeltaM     – n×n perturbation on M = A*B
%   method     – 'DP' | 'GCV' | otherwise: fixed λ
%
% Outputs:
%   x             – computed solution
%   error_norm    – ||x_k–x_true||/||x_true|| at each iterate
%   residual_norm – ||b–A*x_k||/||b|| at each iterate
%   niters        – number of iterations performed
%   phi, dPhi     – filter factors and 1st-order bounds at final k
%   lambda_vec    – λ_k chosen at each iteration

% Noise-level for DP
delta = norm(b - A*x_true);

% Build M and perturbation
M  = A * B;
dK = M' * DeltaM + DeltaM' * M;

% Arnoldi setup
m = size(A,1);
Q = zeros(m, maxit+1);
H = zeros(maxit+1, maxit);
r0 = b - A*(B*zeros(size(B,2),1));
beta = norm(r0);
Q(:,1) = r0 / beta;
e1 = [beta; zeros(maxit,1)];

residual_norm = zeros(maxit,1);
error_norm    = zeros(maxit,1);
lambda_vec    = zeros(maxit,1);

% Main loop
for k = 1:maxit
    % Arnoldi on M = A*B
    v = A*(B*Q(:,k));
    for j = 1:k
        H(j,k) = Q(:,j)' * v;
        v = v - H(j,k) * Q(:,j);
    end
    H(k+1,k) = norm(v);
    if H(k+1,k) == 0, break; end
    Q(:,k+1) = v / H(k+1,k);

    % Projected problem
    Hk = H(1:k+1,1:k);
    tk = e1(1:k+1);

    % Choose lambda_k
    switch upper(method)
        case 'DP'
            dp_fun = @(x) norm(Hk*((Hk'*Hk + x*eye(k))\(Hk'*tk)) - tk) - delta;
            lambda_k = fzero(dp_fun, [1e-16, 1e2]);
        case 'GCV'
            vals = logspace(-16,2,200);
            G = arrayfun(@(x) gcv_fun(x,Hk,tk), vals);
            [~,id] = min(G);
            lambda_k = vals(id);
        otherwise
            lambda_k = lambda;
    end
    lambda_vec(k) = lambda_k;

    % Solve small Tikhonov
    yk = (Hk'*Hk + lambda_k*eye(k)) \ (Hk'*tk);
    uk = Q(:,1:k) * yk;
    xk = B * uk;

    % Record norms, stopping
    rk = b - A*xk;
    residual_norm(k) = norm(rk)/norm(b);
    error_norm(k)    = norm(xk - x_true)/norm(x_true);
    if residual_norm(k) <= tol, break; end
end

% Trim
niters = k;
lambda_vec = lambda_vec(1:k);
residual_norm = residual_norm(1:k);
error_norm    = error_norm(1:k);
x = xk;

% Compute SVD of A
[~,Sa,Va] = svds(A, niters);
sigmaA = diag(Sa);
tilde = sigmaA.^2;
tilde2 = tilde.^2;

% Compute Ritz shifts
Qk = Q(:,1:k);
Hs = H(1:k,1:k);
dKs = Qk'*(dK*Qk);
[Vh,Th] = eig(Hs);
Theta = max(abs(diag(Th)),1e-12);
dTheta = diag(Vh'*(dKs*Vh));

% Compute dMu
[~,SM,VM] = svds(M,niters);
MU = M*VM;
DMV= DeltaM*VM;
dMu = sum(VM.*(M.'*DMV + DeltaM.'*MU),1).';

% Use final lambda
lambda_opt = lambda_vec(end);
s2l = tilde2 + lambda_opt;

% Compute phi & dPhi
Clog = sum(log(max(1 - s2l./Theta.', -1+eps)),2);
P = exp(Clog);
P_excl = exp(Clog - log(max(1 - s2l./Theta', -1+eps)));
phi_z = (tilde2./s2l).*(1-P);
phi   = sigmaA.*phi_z;
term1 = -tilde2 .* sum((dTheta.'./Theta.'.^2).*P_excl,2);
term2 = (lambda_opt./s2l.^2).*(1-P).*dMu;
term3 = (tilde2./s2l).*sum((1./Theta').*P_excl,2).*dMu;
dPhi  = term1 + term2 + term3;
end

% GCV subfunction
function G = gcv_fun(lambda,Hk,tk)
    y = (Hk'*Hk + lambda*eye(size(Hk,2))) \ (Hk'*tk);
    r = Hk*y - tk;
    T = Hk*((Hk'*Hk + lambda*eye(size(Hk,2))) \ Hk');
    G = norm(r)^2 / ((length(tk) - trace(T))^2);
end
