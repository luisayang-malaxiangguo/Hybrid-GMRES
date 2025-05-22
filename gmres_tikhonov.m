function [x, error_norm, residual_norm, niters] = gmres_tikhonov(A, b, x_true, tol, maxit, lambda)
%GMRES_TIKONOV  Hybrid GMRES with Tikhonov regularization
%
%   [x, err, res, niters] = gmres_tikhonov(A, b, x_true, tol, maxit, lambda)
%
%   solves   min_x ||b - A x||^2 + lambda*||x||^2
%   by building the Arnoldi basis Q,H for A and at each k solving
%     min_y || beta*e1 - H_k * y ||^2 + lambda * ||y||^2,
%   then x_k = Q(:,1:k)*y.
%
%   Inputs:
%     A       – n×n matrix or handle
%     b       – right-hand side (n×1)
%     x_true  – exact solution (n×1), for error tracking
%     tol     – tolerance on rel-error ‖x_k - x_true‖/‖x_true‖
%     maxit   – maximum number of GMRES iterations
%     lambda  – regularization parameter
%
%   Outputs:
%     x               – final regularized solution
%     residual_norm   – vector of ‖b - A x_k‖/‖b‖
%     error_norm      – vector of ‖x_k - x_true‖/‖x_true‖
%     niters          – number of iterations performed



    n = size(A,1);
    x = zeros(n,1);

    % initial residual and Arnoldi setup
    r0 = b - A*x; 
    beta = norm(r0);
    Q = zeros(n, maxit+1);
    H = zeros(maxit+1, maxit);
    Q(:,1) = r0/beta;
    e1 = zeros(maxit+1,1);  e1(1) = beta;

    residual_norm = zeros(maxit,1);
    error_norm    = zeros(maxit,1);

    for k = 1:maxit
        %---- Arnoldi step ----
        v = A * Q(:,k);
        for j = 1:k
            H(j,k) = Q(:,j)' * v;
            v       = v - H(j,k)*Q(:,j);
        end
        H(k+1,k) = norm(v);
        if H(k+1,k)==0
            break;
        end
        Q(:,k+1) = v / H(k+1,k);

        % Tikhonov solve in k-dimensional Hessenberg 
        Hk = H(1:k+1, 1:k);
        c  = e1(1:k+1);           % = [beta; 0; …; 0]
        % solve  (Hk'*Hk + lambda*I) y = Hk' * c. normal eqs for tik prob
        M  = Hk'*Hk + lambda*eye(k);
        rhs= Hk' * c;
        yk = M \ rhs;

        % xk=Qk*yk
        xk = Q(:,1:k) * yk;

        % track norms
        rk = b - A*xk;
        residual_norm(k) = norm(rk)/norm(b);
        error_norm(k)    = norm(xk - x_true)/norm(x_true);

        if error_norm(k) <= tol
            break;
        end
    end

    % trim outputs
    niters = k;
    residual_norm = residual_norm(1:k);
    error_norm    = error_norm(1:k);
    x = xk;
end
