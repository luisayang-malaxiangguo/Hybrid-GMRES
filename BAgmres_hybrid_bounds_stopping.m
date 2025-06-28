function [x, error_norm, residual_norm, niters, phi, dPhi, lambda_vec] = BAgmres_hybrid_bounds_stopping( ...
    A, B, b, x_true, tol, maxit, lambda, DeltaM, method)
% BAGMRES_HYBRID_BOUNDS_STOPPING  BA-GMRES with Tikhonov + DP/GCV-based λ + 1st-order bounds
%
%   [x, err, res, k, phi, dPhi, lambda_vec] = ...
%     BAgmres_hybrid_bounds_stopping(A,B,b,x_true,tol,maxit,lambda,DeltaM,method)
%
% Inputs:
%   A, B       – forward operator (m×n) and back-projector (n×m)
%   b, x_true  – data and true solution (for DP noise estimate)
%   tol        – stopping tolerance on ||b–A*x||/||b||
%   maxit      – maximum GMRES iterations
%   lambda     – fallback Tikhonov parameter (if method~='DP'/'GCV')
%   DeltaM     – n×n perturbation on M = B*A
%   method     – 'DP' | 'GCV' | otherwise: fixed λ
%
% Outputs:
%   x             – computed solution
%   error_norm    – ||x_k–x_true||/||x_true|| at each iterate
%   residual_norm – ||b–A*x_k||/||b|| at each iterate
%   niters        – number of iterations performed (k)
%   phi, dPhi     – length-k vectors of unperturbed filter factors
%                   and their first-order perturbation bounds
%   lambda_vec    – λ_k chosen at each iteration

  % Estimate noise–level for discrepancy principle
  delta = norm(b - A*x_true);

  % Build M and its perturbation ΔK
  M  = B * A;
  dK = M' * DeltaM + DeltaM' * M;

  %— Arnoldi setup on BA —%
  n    = size(A,2);
  x0   = zeros(n,1);
  r0   = B * (b - A*x0);
  beta = norm(r0);

  Q = zeros(n, maxit+1);
  H = zeros(maxit+1, maxit);
  Q(:,1) = r0 / beta;
  e1     = [beta; zeros(maxit,1)];

  residual_norm = zeros(maxit,1);
  error_norm    = zeros(maxit,1);
  lambda_vec    = zeros(maxit,1);

  %— Main GMRES loop —%
  for k = 1:maxit
    % Arnoldi step on M = B*A
    v = B * (A * Q(:,k));
    for j = 1:k
      H(j,k) = Q(:,j)' * v;
      v       = v - H(j,k) * Q(:,j);
    end
    H(k+1,k) = norm(v);
    if H(k+1,k) == 0, break; end
    Q(:,k+1) = v / H(k+1,k);

    % Projected problem
    Hk = H(1:k+1,1:k);
    tk = e1(1:k+1);

    % Choose λ_k
    switch upper(method)
      case 'DP'
        dp_fun   = @(x) norm(Hk*((Hk'*Hk + x*eye(k))\(Hk'*tk)) - tk) - delta;
        lambda_k = fzero(dp_fun, [1e-16, 1e2]);
      case 'GCV'
        lambdas  = logspace(-16, 2, 200);
        G        = arrayfun(@(x) gcv_fun(x, Hk, tk), lambdas);
        [~, idx] = min(G);
        lambda_k = lambdas(idx);
      otherwise
        lambda_k = lambda;
    end
    lambda_vec(k) = lambda_k;

    % Tikhonov solve with λ_k
    yk = (Hk'*Hk + lambda_k*eye(k)) \ (Hk'*tk);
    xk = Q(:,1:k) * yk;

    % Norms & stopping
    rk = b - A*xk;
    residual_norm(k) = norm(rk)/norm(b);
    error_norm(k)    = norm(xk - x_true)/norm(x_true);
    if residual_norm(k) <= tol, break; end
  end

  % Trim outputs to actual iteration count
  niters        = k;
  lambda_vec    = lambda_vec(1:k);
  x              = xk;
  residual_norm = residual_norm(1:k);
  error_norm    = error_norm(1:k);

  %— Compute unperturbed φ and dPhi at final k —%
  [~, S, V] = svds(M, niters);
  sigma     = diag(S);

  % Ritz‐values + perturbations
  Qk      = Q(:,1:k);
  Hs      = H(1:k,1:k);
  dK_s    = Qk' * (dK * Qk);
  [Vh, Th]= eig(Hs);
  Theta   = max(abs(diag(Th)), 1e-12);
  dTheta  = diag(Vh' * (dK_s * Vh));

  % Singular‐value perturbations
  MU   = M      * V;
  DMV  = DeltaM * V;
  dMu  = sum(V .* (M.'*DMV + DeltaM.'*MU), 1).';

  % Use final λ
  lambda_opt = lambda_vec(end);
  s2l        = sigma.^2 + lambda_opt;

  Clog   = sum(log(max(1 - s2l./Theta.', -1+1e-16)), 2);
  P      = exp(Clog);
  P_excl = exp( Clog - log(max(1 - s2l./Theta', -1+1e-16)) );

  phi    = (sigma.^2 ./ s2l) .* (1 - P);
  term1  = -sigma.^2 .* sum((dTheta.' ./ Theta.'.^2) .* P_excl, 2);
  term2  =  (lambda_opt ./ s2l.^2)    .* (1 - P)       .* dMu;
  term3  =  (sigma.^2     ./ s2l)      .* sum((1./Theta') .* P_excl,2) .* dMu;
  dPhi   = term1 + term2 + term3;
end

%------------------------------------------------------------------------------%
function G = gcv_fun(lambda, Hk, tk)
  k = size(Hk,2);
  y = (Hk'*Hk + lambda*eye(k)) \ (Hk'*tk);
  r = Hk*y - tk;
  T = Hk * ((Hk'*Hk + lambda*eye(k)) \ Hk');
  G = norm(r)^2 / ((length(tk) - trace(T))^2);
end
