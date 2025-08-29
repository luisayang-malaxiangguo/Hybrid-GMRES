function [x, error_norm, residual_norm, niters] = hybrid_ba_gmres_rtp(A, B, b, x_true, tol, maxit, lambda) 

    n = size(A, 2);
    x = zeros(n, 1);
     
    M_reg = @(v) B*(A*v) + lambda*v;
    d = B * b;
     
    r0 = d - M_reg(x);
    beta = norm(r0);
    Q = zeros(n, maxit + 1);
    H = zeros(maxit + 1, maxit);
    Q(:, 1) = r0 / beta;
    
    error_norm = zeros(maxit, 1);
    residual_norm = zeros(maxit, 1);

    for k = 1:maxit 
        v = M_reg(Q(:, k));
        for j = 1:k
            H(j, k) = Q(:, j)' * v;
            v = v - H(j, k) * Q(:, j);
        end
        H(k + 1, k) = norm(v);
        if H(k + 1, k) == 0, break; end
        Q(:, k + 1) = v / H(k + 1, k);

        Hk = H(1:k + 1, 1:k);
        yk = Hk \ ([beta; zeros(k, 1)]);
        x = Q(:, 1:k) * yk;
         
        residual_norm(k) = norm(b - A*x) / norm(b);
        error_norm(k) = norm(x - x_true) / norm(x_true);

        if residual_norm(k) <= tol, break; end
    end

    niters = k;
    residual_norm = residual_norm(1:k);
    error_norm = error_norm(1:k);

end

