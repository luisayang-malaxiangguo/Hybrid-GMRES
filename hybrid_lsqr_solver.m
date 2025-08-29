function [x, error_norm, residual_norm, niters] = hybrid_lsqr_solver(A, b, x_true, tol, maxit, lambda)

    [m, n] = size(A);
      
    A_aug = [A; sqrt(lambda) * eye(n)];
    b_aug = [b; zeros(n, 1)];
     
    x = zeros(n, 1);
    beta_aug = norm(b_aug);
    u_aug = b_aug / beta_aug;
    v_hat = A_aug' * u_aug;
    alpha_aug = norm(v_hat);
    v = v_hat / alpha_aug;
    w = v;
    phi_bar = beta_aug;
    rho_bar = alpha_aug;
    
    error_norm = zeros(maxit, 1);
    residual_norm = zeros(maxit, 1);
    
    for k = 1:maxit
        u_hat = A_aug * v - alpha_aug * u_aug;
        beta_aug = norm(u_hat);
        u_aug = u_hat / beta_aug;
        
        v_hat = A_aug' * u_aug - beta_aug * v;
        alpha_aug = norm(v_hat);
        v = v_hat / alpha_aug;
        
        rho = sqrt(rho_bar^2 + beta_aug^2);
        c = rho_bar / rho;
        s = beta_aug / rho;
        
        theta = s * alpha_aug;
        rho_bar = -c * alpha_aug;
        phi = c * phi_bar;
        phi_bar = s * phi_bar;
        
        x = x + (phi / rho) * w;
        w = v - (theta / rho) * w;
         
        error_norm(k) = norm(x - x_true) / norm(x_true);
        residual_norm(k) = norm(b - A*x) / norm(b);
        
        if residual_norm(k) < tol, break; end
    end
    
    niters = k;
    error_norm = error_norm(1:k);
    residual_norm = residual_norm(1:k);

end
