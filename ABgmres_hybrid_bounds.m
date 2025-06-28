function [x, error_norm, residual_norm, niters, phi, dPhi] = ABgmres_hybrid_bounds(A, B, b, x_true, tol, maxit, lambda, DeltaM)
% ABGMRES_HYBRID_BOUNDS  AB-GMRES with Tikhonov + first-order perturbation bounds
%   Additional outputs:
%     phi  — filter factors at final iteration (size = n)
%     dPhi — first-order bound on perturbation of phi (size = n)

    % Build M = A*B and its perturbation ΔK = M'ΔM + ΔM'M
  M  = A*B;
  dK = M' * DeltaM + DeltaM' * M;

  %— Arnoldi + Tikhonov —  
  m = size(A,1);
  u0 = zeros(size(B,2),1);
  x0 = B*u0;
  r0 = b - A*x0;
  beta = norm(r0);

  Q = zeros(m,maxit+1);
  H = zeros(maxit+1,maxit);
  Q(:,1) = r0/beta;
  e1     = [beta; zeros(maxit,1)];
  residual_norm = zeros(maxit,1);
  error_norm    = zeros(maxit,1);

  for k = 1:maxit
    % Arnoldi step on M = A*B
    v = A*(B*Q(:,k));
    for j = 1:k
      H(j,k) = Q(:,j)' * v;
      v       = v - H(j,k) * Q(:,j);
    end
    H(k+1,k) = norm(v);
    if H(k+1,k)==0, break; end
    Q(:,k+1) = v / H(k+1,k);

    % Tikhonov in Krylov subspace
    Hk = H(1:k+1,1:k);    tk = e1(1:k+1);
    yk = (Hk'*Hk + lambda*eye(k)) \ (Hk'*tk);
    uk = Q(:,1:k) * yk;
    xk = B * uk;

    % norms & stopping
    rk = b - A*xk;
    residual_norm(k) = norm(rk)/norm(b);
    error_norm(k)    = norm(xk - x_true)/norm(x_true);
    if residual_norm(k) <= tol, break; end
  end

  niters        = k;
  x              = xk;
  residual_norm = residual_norm(1:k);
  error_norm    = error_norm(1:k);

   %— SVD of A to get singular values σ^A_i —%
  [~, Sa, Va] = svds(A, niters);
  sigmaA      = diag(Sa);      % k×1
  tilde       = sigmaA.^2;     % (σ^A_i)^2
  tilde2      = tilde.^2;      % (σ^A_i)^4

  %— Compute Ritz-value shifts dTheta —%
  Qk       = Q(:,1:k);
  Hk_small = H(1:k,1:k);              % = Qk' * M * Qk
  dK_small = Qk' * (dK * Qk);         % = Qk'*ΔK*Qk
  [Vh, Th] = eig(Hk_small);
  Theta_raw= diag(Th);                % may include near-zero
  theta_min= 1e-12;
  Theta    = max(abs(Theta_raw), theta_min);  % threshold

  dTheta   = diag(Vh' * (dK_small * Vh));     % k×1

  %— Compute singular-value shifts dMu —%
  [~, SM, VM] = svds(M, niters);
  MU   = M      * VM;
  DMV  = DeltaM * VM;
  dMu  = sum(VM .* (M.'*DMV + DeltaM.'*MU), 1).';  % k×1

  %— Stable log-domain product for P = ∏ (1 - s2l/Theta) —%
  s2l = tilde2 + lambda;       % k×1
  Clog = zeros(k,1);
  for i = 1:k
    % clamp argument of log to avoid log(0) or negative
    terms = max(1 - s2l(i)./Theta.', -1+eps);
    Clog(i) = sum(log(terms));
  end
  P = exp(Clog);               % k×1

  % P_excl(i,j) = P(i) / (1 - s2l(i)/Theta(j)), but compute via logs
  P_excl = zeros(k,k);
  for i = 1:k
    for j = 1:k
      denom = max(1 - s2l(i)/Theta(j), -1+eps);
      P_excl(i,j) = exp(Clog(i) - log(denom));
    end
  end

  %— Unperturbed filter φ and perturbed bound dPhi —%
  phi_z = (tilde2 ./ s2l) .* (1 - P);  % z-filter
  phi   = sigmaA .* phi_z;             % x-filter

  term1 = - tilde2 .* sum((dTheta.' ./ Theta.'.^2) .* P_excl, 2);
  term2 =   (lambda   ./ s2l.^2)    .* (1 - P)        .* dMu;
  term3 =     (tilde2 ./ s2l)       .* sum((1./Theta') .* P_excl,2) .* dMu;
  dPhi  = term1 + term2 + term3;
end