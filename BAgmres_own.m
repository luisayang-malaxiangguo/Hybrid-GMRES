function [x, error_norm, residual_norm, niters] = BAgmres_own(BA, b, x_true,tol, maxit)
%BA-GMRES_OWN  Solve BA*x = b using GMRES on the normal‐equations form
%
%   [x, error_norm, residual_norm, niters] =
%       BAgmres_own(BA, b, tol, maxit)
%
%   Inputs:
%     BA      – square (n×n) matrix (A'*A)
%     b       – RHS vector (A'*b)
%     tol     – relative residual tolerance (default 1e-6)
%     maxit   – maximum # of GMRES iterations (default n)
%
%   Outputs:
%     x               – computed solution
%     residual_norm   – ‖b – BA*x_k‖/‖b‖ at each iteration k
%     error_norm      – ‖x_k – x_true‖/‖x_true‖ at each iteration
%                       (with x_true = BA\b)
%     niters          – number of iterations performed


    %— default parameters —
    if nargin < 3 || isempty(tol)
        tol = 1e-6;
    end
    if nargin < 4 || isempty(maxit)
        maxit = size(BA,1);
    end

    % compute “true” solution for error tracking
    %x_true = BA \ b;

    n = size(BA,1);
    x = zeros(n,1);

    % initial residual
    r0 = b - BA*x;
    beta = norm(r0);

    % allocate Arnoldi basis Q and Hessenberg H
    Q = zeros(n, maxit+1);
    H = zeros(maxit+1, maxit);

    Q(:,1) = r0 / beta;
    e1 = zeros(maxit+1,1);
    e1(1) = beta;

    residual_norm = zeros(maxit,1);
    error_norm    = zeros(maxit,1);

    %— Arnoldi + GMRES iterations —
    for k = 1:maxit
        % Arnoldi step on BA
        q = BA * Q(:,k);
        for i = 1:k
            H(i,k) = Q(:,i)' * q;
            q      = q - H(i,k)*Q(:,i);
        end
        H(k+1,k) = norm(q);
        if H(k+1,k) == 0
            break
        end
        Q(:,k+1) = q / H(k+1,k);

        % solve the small least‐squares problem
        Hk = H(1:k+1,1:k);
        yk  = Hk \ e1(1:k+1);

        % reconstruct x_k = Q(:,1:k)*y
        xk = Q(:,1:k) * yk;

        % compute relative norms
        rk = b - BA*xk;
        residual_norm(k) = norm(rk) / norm(b);
        error_norm(k)    = norm(xk - x_true) / norm(x_true);

        % check convergence
        if residual_norm(k) <= tol
            break
        end
    end

    % trim outputs and return
    niters        = k;
    residual_norm = residual_norm(1:k);
    error_norm    = error_norm(1:k);
    x             = xk;
end
