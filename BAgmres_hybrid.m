function [x, error_norm, residual_norm, niters] = BAgmres_hybrid(A, B, b, x_true, tol, maxit, lambda)
%BA_GMRES_HYBRID   BA-GMRES with Tikhonov regularization
%
%   [x, error_norm, residual_norm, niters] =
%       BAgmres_hybrid(A, B, b, x_true, tol, maxit, lambda)
%
%   Solves the system BA*x = B*b using GMRES on BA, incorporating Tikhonov
%   regularization in the small projected subproblem at each iteration.  At
%   iteration k, it solves
%       min_y ||beta*e1 - H_k*y||^2 + lambda*||y||^2,
%   then sets x_k = Q_k*y.
%
%   Inputs:
%     A       – m×n forward operator
%     B       – n×m left-preconditioner (e.g., back projector)
%     b       – right-hand side (m×1)
%     x_true  – exact solution (n×1) for error tracking
%     tol     – relative tolerance on residual ||b - A*x||/||b||
%     maxit   – maximum GMRES iterations
%     lambda  – Tikhonov regularization parameter
%
%   Outputs:
%     x               – computed solution (n×1)
%     residual_norm   – vector of ||b - A*x_k||/||b||
%     error_norm      – vector of ||x_k - x_true||/||x_true||
%     niters          – number of iterations performed

 

    % Initial solution and residual in x-space
    n = size(A,2);
    x0 = zeros(n,1);
    r0 = B * (b - A*x0);
    beta = norm(r0);

    % Allocate Arnoldi basis Q and Hessenberg H
    Q = zeros(n, maxit+1);
    H = zeros(maxit+1, maxit);
    Q(:,1) = r0 / beta;
    e1 = zeros(maxit+1,1);
    e1(1) = beta;

    residual_norm = zeros(maxit,1);
    error_norm    = zeros(maxit,1);

    % GMRES loop with Tikhonov in projected problem
    for k = 1:maxit
        %--- Arnoldi on BA ---
        v = B * (A * Q(:,k));
        for j = 1:k
            H(j,k) = Q(:,j)' * v;
            v       = v - H(j,k)*Q(:,j);
        end
        H(k+1,k) = norm(v);
        if H(k+1,k) == 0
            break;
        end
        Q(:,k+1) = v / H(k+1,k);

        %--- Tikhonov solve in small space ---
        Hk   = H(1:k+1, 1:k);
        tk   = e1(1:k+1);
        M    = Hk' * Hk + lambda * eye(k);
        rhs  = Hk' * tk;
        yk   = M \ rhs;

        %--- Update x_k ---
        xk = Q(:,1:k) * yk;

        %--- Compute norms and check ---
        rk = b - A*xk;
        residual_norm(k) = norm(rk) / norm(b);
        error_norm(k)    = norm(xk - x_true) / norm(x_true);
        if residual_norm(k) <= tol
            break;
        end
    end

    % Trim outputs
    niters        = k;
    residual_norm = residual_norm(1:k);
    error_norm    = error_norm(1:k);
    x             = xk;
end
