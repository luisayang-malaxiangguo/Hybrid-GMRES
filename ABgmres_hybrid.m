function [x, error_norm, residual_norm, niters] = ABgmres_hybrid(A, B, b, x_true, tol, maxit, lambda)
%ABGMRES_HYBRID  AB‑GMRES with Tikhonov regularization (Arnoldi–Tikhonov)
%
%   [x, error_norm, residual_norm, niters] = \
%       ABgmres_hybrid(A, B, b, x_true, tol, maxit, lambda)
%
%   Solves the right‑preconditioned system AB*u = b with x = B*u,
%   applying Tikhonov regularization in the Krylov subspace at each
%   iteration.  At step k it solves
%       min_y ||beta*e1 - H_k*y||^2 + lambda*||y||^2,
%   then sets u_k = Q_k*y, x_k = B*u_k.
%
%   Inputs:
%     A       – m×n forward operator (matrix or function handle)
%     B       – n×m back‑projector / preconditioner
%     b       – right‑hand side (m×1)
%     x_true  – exact solution (n×1), for error tracking
%     tol     – stopping tolerance on relative residual ||b-Ax||/||b||
%     maxit   – maximum number of iterations
%     lambda  – Tikhonov regularization parameter
%
%   Outputs:
%     x               – computed solution (n×1)
%     residual_norm   – vector of ||b - A*x_k||/||b||
%     error_norm      – vector of ||x_k - x_true||/||x_true||
%     niters          – number of iterations performed

   

    m = size(A,1);
    % initial u0=0 => x0 = B*u0 = 0
    u0 = zeros(size(B,2),1);
    x0 = B * u0;
    r0 = b - A * x0;
    beta = norm(r0);

    % allocate Arnoldi basis and Hessenberg
    Q = zeros(m, maxit+1);
    H = zeros(maxit+1, maxit);
    Q(:,1) = r0 / beta;
    e1 = zeros(maxit+1,1);
    e1(1) = beta;

    residual_norm = zeros(maxit,1);
    error_norm    = zeros(maxit,1);

    for k = 1:maxit
        %--- Arnoldi on AB ---
        v = A * (B * Q(:,k));
        for j = 1:k
            H(j,k) = Q(:,j)' * v;
            v       = v - H(j,k) * Q(:,j);
        end
        H(k+1,k) = norm(v);
        if H(k+1,k) == 0
            break;
        end
        Q(:,k+1) = v / H(k+1,k);

        %--- Tikhonov solve in small space ---
        Hk  = H(1:k+1, 1:k);
        tk  = e1(1:k+1);
        % normal equations: (Hk' * Hk + lambda*I) y = Hk' * tk
        M    = Hk' * Hk + lambda * eye(k);
        rhs  = Hk' * tk;
        yk   = M \ rhs;

        %--- back-transform to full space ---
        uk = Q(:,1:k) * yk;
        xk = B * uk;

        %--- norms and convergence ---
        rk = b - A * xk;
        residual_norm(k) = norm(rk) / norm(b);
        error_norm(k)    = norm(xk - x_true) / norm(x_true);
        if residual_norm(k) <= tol
            break;
        end
    end

    % trim outputs
    niters = k;
    residual_norm = residual_norm(1:k);
    error_norm    = error_norm(1:k);
    x = xk;
end
