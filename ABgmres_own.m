function [x, error_norm, residual_norm, niters] = ABgmres_own(A, B, b, x_true, tol, maxit)
%ABGMRES_OWN  Solve A*x = b using AB-GMRES (right-preconditioned GMRES)
%
%   [x, error_norm, residual_norm, niters] =
%       ABgmres_own(A, B, b, tol, maxit)
%
%   Inputs:
%     A       – m×n matrix
%     B       – n×m right preconditioner (so that x = B*u)
%     b       – right-hand side (m-vector)
%     tol     – relative tolerance on the residual
%     maxit   – maximum number of GMRES iterations
%
%   Outputs:
%     x               – computed solution (n-vector)
%     residual_norm   – ‖b – A*x_k‖/‖b‖ at each iteration k
%     error_norm      – ‖x_k – x_true‖/‖x_true‖ at each iteration
%                       (requires x_true = A\b, computed internally)
%     niters          – number of iterations actually performed

    %----- set up and defaults -----
    if nargin<4 || isempty(tol),    tol    = 1e-6;        end
    if nargin<5 || isempty(maxit),  maxit  = size(A,1);   end

    % compute exact solution for error tracking
    %x_true = A \ b;

    % initial u‐vector is zero → x0 = B*u0 = 0
    m = size(A,1);
    u0 = zeros(size(B,2),1);
    x0 = B * u0;

    % initial residual
    r0 = b - A*x0;
    beta = norm(r0);

    % allocate Arnoldi basis in the space of u (dimension = length(u0))
    Q = zeros(m, maxit+1);      % basis for Krylov of AB
    H = zeros(maxit+1, maxit);  % Hessenberg matrix

    Q(:,1) = r0 / beta;
    e1 = zeros(maxit+1,1);  
    e1(1) = beta;

    residual_norm = zeros(maxit,1);
    error_norm    = zeros(maxit,1);

    %----- Arnoldi + GMRES loop -----
    for k = 1:maxit
        % apply the AB operator: w = A*(B * v_k)
        q = A* (B * Q(:,k));

        % classical Gram‐Schmidt
        for i = 1:k
            H(i,k) = Q(:,i)' * q;
            q = q - H(i,k)*Q(:,i);
        end
        H(k+1,k) = norm(q);
        if H(k+1,k) == 0
            break
        end
        Q(:,k+1) = q / H(k+1,k);

        % solve small least‐squares problem
        Hk = H(1:k+1,1:k);
        yk  = Hk \ e1(1:k+1);

        % reconstruct u_k and then x_k
        uk = Q(:,1:k) * yk; % V_k*y_k
        xk = B * uk; %x_k=B*u_k

        % compute norms
        rk = b - A*xk;
        residual_norm(k) = norm(rk) / norm(b);
        error_norm(k)    = norm(xk - x_true) / norm(x_true);

        % check convergence
        if residual_norm(k) <= tol
            break
        end
    end

    % trim to actual iteration count
    niters        = k;
    residual_norm = residual_norm(1:k);
    error_norm    = error_norm(1:k);
    x             = xk;
end
