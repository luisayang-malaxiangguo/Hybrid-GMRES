function [x, error_norm, residual_norm, niters] = hybrid_ab_gmres_rtp(A, B, b, x_true, tol, maxit, lambda)
% HYBRID_AB_GMRES_RTP Regularize-then-Project version of Hybrid AB-GMRES.
% The iterate x_k lies in K_k(BA + lambda*I, Bb) and solves min{||Ax-b||^2 + lambda*||x||^2}.

    n = size(A, 2);
    x0 = zeros(n, 1);

    % The Krylov subspace is K_k(BA + lambda*I, Bb)
    M_reg_op = @(v) B*(A*v) + lambda*v;
    d_krylov = B*b;
    
    r0 = d_krylov - M_reg_op(x0);
    beta = norm(r0);
    Q = zeros(n, maxit + 1);
    H = zeros(maxit + 1, maxit);  
    Q(:, 1) = r0 / beta;

    error_norm = zeros(maxit, 1);
    residual_norm = zeros(maxit, 1);

    for k = 1:maxit 
        v = M_reg_op(Q(:, k));
        for j = 1:k
            H(j, k) = Q(:, j)' * v;
            v = v - H(j, k) * Q(:, j);
        end
        H(k + 1, k) = norm(v);
        if H(k + 1, k) == 0, break; end
        Q(:, k + 1) = v / H(k + 1, k);
        
        Qk = Q(:, 1:k);
        
        %  Solve the Tikhonov problem projected onto that subspace 
        % min ||A*Qk*y - b||^2 + lambda*||Qk*y||^2
        % This is equivalent to solving for y in:
        % ((AQk)'(AQk) + lambda*I) y = (AQk)' b
        
        AQk = A * Qk;
        yk = (AQk' * AQk + lambda * eye(k)) \ (AQk' * b);
        x = Qk * yk;
  
        residual_norm(k) = norm(b - A*x) / norm(b);
        error_norm(k) = norm(x - x_true) / norm(x_true);

        if residual_norm(k) <= tol, break; end
    end
    
    niters = k;
    residual_norm = residual_norm(1:k);
    error_norm = error_norm(1:k);
end