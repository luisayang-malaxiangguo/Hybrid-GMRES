function [x, error_norm, residual_norm, niters] = lsqr_solver(A, b, x_true, tol, maxit)
% LSQR_SOLVER Solvesmin||Ax-b||.

    % Initialization
    m = size(A, 1);
    n = size(A, 2);
    x = zeros(n, 1);
    
    beta = norm(b);
    u = b / beta;
    
    v_hat = A' * u;
    alpha = norm(v_hat);
    v = v_hat / alpha;
    
    w = v;
    phi_bar = beta;
    rho_bar = alpha;
    
    error_norm = zeros(maxit, 1);
    residual_norm = zeros(maxit, 1);
    for k = 1:maxit
        %  GKB Step 
        u_hat = A * v - alpha * u;
        beta = norm(u_hat);
        u = u_hat / beta;
        
        v_hat = A' * u - beta * v;
        alpha = norm(v_hat);
        v = v_hat / alpha;
        
        %  Givens Rotation 
        rho = sqrt(rho_bar^2 + beta^2);
        c = rho_bar / rho;
        s = beta / rho;
        
        theta = s * alpha;
        rho_bar = -c * alpha;
        phi = c * phi_bar;
        phi_bar = s * phi_bar;
         
        x = x + (phi / rho) * w;
        w = v - (theta / rho) * w;
         
        error_norm(k) = norm(x - x_true) / norm(x_true);
        residual_norm(k) = abs(phi_bar) / norm(b);  
        
        if residual_norm(k) <= tol, break; end
    end
    
    niters = k;
    error_norm = error_norm(1:k);
    residual_norm = residual_norm(1:k); 
    residual_norm(end) = norm(b-A*x)/norm(b);
end