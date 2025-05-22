
%% Tomography example
n = 64;
[A, b, x_true, ProbInfo] = PRtomo(n); % x is exact solution A\b

imagesc(reshape(x_true,ProbInfo.xSize))
imagesc(reshape(b,ProbInfo.bSize))
%%
tol  = 1e-6;
maxit = 100;
%% Tomography example
% Try GMRES
Agmres = A'*A;
bgmres = A'*b;
[x_iterates, error_norm, residual_norm, niters] = gmres_own(Agmres, bgmres,x_true, tol, maxit);


% iteration vector
iters = 1:niters;

% semilogy plot
figure(1);
semilogy(iters, residual_norm, 'o-', 'LineWidth', 1.5); hold on;
semilogy(iters, error_norm,    's-', 'LineWidth', 1.5);
hold off;


xlabel('Iteration k');
ylabel('Relative norm');
legend('||b - A x_k||/||b||','||x_k - x_{true}||/||x_{true}||','Location','best');
title('GMRES Convergence History');
grid on;


%%
% Try AB-GMRES
A = A;
B = A';
AB=A*B;
[x_iterates, error_norm, residual_norm, niters] = ABgmres_own(A, B, b, x_true, tol, maxit);


figure(2);
semilogy(1:niters, residual_norm, 'o-', 1:niters, error_norm, 's-');
xlabel('Iteration k');
ylabel('Relative norm');
legend('||b - A x_k||/||b||', '||x_k - x_{true}||/||x_{true}||','Location','best');
title('AB-GMRES Convergence History');
grid on;

%%
% Try BA-GMRES
A = A;
B = A';
BA=B*A;
[x_iterates, error_norm, residual_norm, niters] = BAgmres_own(BA, B*b,x_true, tol, maxit);


figure(3);
semilogy(1:niters, residual_norm, 'o-', 1:niters, error_norm, 's-');
xlabel('Iteration k');
ylabel('Relative norm');
legend('||b - A x_k||/||b||', '||x_k - x_{true}||/||x_{true}||','Location','best');
title('BA-GMRES Convergence History');
grid on;

