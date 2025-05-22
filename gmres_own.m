function [x, error_norm, residual_norm, niters] = gmres_own(A, b, x_true, tol, maxit)
%GMRES_OWN  Solve A'A*x = A'b using GMRES
%
%   [x, error_norm, residual_norm] =
%     gmres_own(A, b, tol, maxit)
%
%   Inputs:
%     A       – square (n×n) matrix or function handle
%     b       – right-hand side (n-vector)
%     x_true   – the exact solution (n-vector), for error_norm
%     tol     – relative tolerance (default 1e-6)
%     maxit   – maximum # of iterations (default n)
%
%   Outputs:
%     x               – computed solution
%     residual_norm   – residual norms ‖b–A*x_k‖ at each iteration (if want relative then
%     divide by ‖b‖)
%     error_norm      – residual norms ‖x_true-x_k‖ (if want relative then
%     divide by ‖x_true‖)

%    if nargin < 3 || isempty(tol)  
%       tol    = 1e-6;    
%   end
%    if nargin < 4 || isempty(maxit)
%        maxit  = size(A,1);
%    end

% can remove lines 21--26 if we input manually tol and maxtit


    n = size(A,1);
    x = zeros(n,1);
    % initial residual
    r0 = b - A*x;
    beta = norm(r0);

    % Arnoldi basis Q and Hessenberg
    Q = zeros(n, maxit+1); %have q_{k+1} vectors, k iterations
    H = zeros(maxit+1, maxit); %H:(k+1,k), k iterations

    Q(:,1) = r0 / beta; %q_1=r_0/||r_0||
    e1 = zeros(maxit+1,1);  e1(1) = beta; %used later to solve least square

    residual_norm = zeros(maxit,1);
    error_norm    = zeros(maxit,1);

    for k = 1:maxit
        %--- Arnoldi step ---
        q = A * Q(:,k);
        for i = 1:k
            H(i,k) = Q(:,i)' * q;
            q = q - H(i,k)*Q(:,i);
        end
        H(k+1,k) = norm(q);
        if H(k+1,k)==0
            break
        end
        Q(:,k+1) = q / H(k+1,k);

        %--- Solve least-squares H(1:k+1,1:k)*y ≈ beta*e1 ---
        Hk = H(1:k+1,1:k);  
        yk  = Hk \ e1(1:k+1); %y=Hk\(norm(r0),0...,0)'
        xk = Q(:,1:k) * yk; %xk=Q_k*y

        %--- Compute norms ---
        rk = b - A*xk;
        residual_norm(k) = norm(rk)/norm(b);
        error_norm(k)    = norm(xk - x_true)/norm(x_true);

        %--- Check convergence ---
        if error_norm(k) <= tol
            break
        end
    end

    % Trim to actual iterations
    residual_norm = residual_norm(1:k);
    error_norm    = error_norm(1:k);
    x = xk;
    niters = k;
end
