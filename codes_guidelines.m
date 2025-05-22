%% Tomography example
n = 64;
[A, b, x_true, ProbInfo] = PRtomo(n); % x_true is exact solution A\b

imagesc(reshape(x_true,ProbInfo.xSize))
imagesc(reshape(b,ProbInfo.bSize))

[bn, NoiseInfo] = PRnoise(b, 0.01);
tol  = 1e-6;
maxit = 100;
%% Tomography example
% GMRES
Agmres = A'*A;
bgmres = A'*b;
[x_iterates, error_norm, residual_norm, niters] = gmres_own(A'*A, A'*b, x_true, tol, maxit);

figure(1);
semilogy(1:niters, residual_norm, 'o-',1:niters, error_norm,'s-');
xlabel('Iteration k');
ylabel('Relative norm');
legend('||b - A x_k||/||b||','||x_k - x_{true}||/||x_{true}||','Location','best');
title('GMRES Convergence History');
grid on;


%% AB-GMRES
A = A;
B = A';
[x_iterates, error_norm, residual_norm, niters] = ABgmres_own(A, B, bn, x_true, tol, maxit);


figure(2);
semilogy(1:niters, residual_norm, 'o-', 1:niters, error_norm, 's-');
xlabel('Iteration k');
ylabel('Relative norm');
legend('||b_{noise} - A x_k||/||b_{noise}||', '||x_k - x_{true}||/||x_{true}||','Location','best');
title('AB-GMRES Convergence History');
grid on;

%% BA-GMRES
A = A;
B = A';
[x_iterates, error_norm, residual_norm, niters] = BAgmres_own(A, B, bn, x_true, tol, maxit);


figure(3);
semilogy(1:niters, residual_norm, 'o-', 1:niters, error_norm, 's-');
xlabel('Iteration k');
ylabel('Relative norm');
legend('||b_{noise} - A x_k||/||b_{noise}||', '||x_k - x_{true}||/||x_{true}||','Location','best');
title('BA-GMRES Convergence History');
grid on;

%% B approx to A'
A = A;
tau=0.5;
T = tau* max(A(:));        % threshold = max over all entries of A
B = A.';               % swap indices: B(i,j) = A(j,i)
B(B < T) = 0;  

lambda=5;

%% GMRES-Tikhonov


[x_iterates, error_norm, residual_norm, niters] = gmres_tikhonov(A'*A, A'*bn, x_true, tol, maxit, lambda);

figure(4);
semilogy(1:niters, residual_norm, 'o-', 1:niters, error_norm, 's-');
xlabel('Iteration k');
ylabel('Relative norm');
legend('||b_{noise} - A x_k||/||b_{noise}||', '||x_k - x_{true}||/||x_{true}||','Location','best');
title('GMRES Tikhonov Convergence with B \approx A^T');
grid on;





%% AB-GMRES Tikhonov


[x_iterates, error_norm, residual_norm, niters] = ABgmres_hybrid(A, B, bn, x_true, tol, maxit, lambda);

figure(5);
semilogy(1:niters, residual_norm, 'o-', 1:niters, error_norm, 's-');
xlabel('Iteration k');
ylabel('Relative norm');
legend('||b_{noise} - A x_k||/||b_{noise}||', '||x_k - x_{true}||/||x_{true}||','Location','best');
title('AB-GMRES Tikhonov Convergence with B \approx A^T');
grid on;

%% BA-GMRES Tikhonov


[x_iterates, error_norm, residual_norm, niters] = BAgmres_hybrid(A, B, bn, x_true, tol, maxit, lambda);

figure(6);
semilogy(1:niters, residual_norm, 'o-', 1:niters, error_norm, 's-');
xlabel('Iteration k');
ylabel('Relative norm');
legend('||b_{noise} - A x_k||/||b_{noise}||', '||x_k - x_{true}||/||x_{true}||','Location','best');
title('BA-GMRES Tikhonov Convergence with B \approx A^T');
grid on;