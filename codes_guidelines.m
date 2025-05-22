%% Tomography example
n = 64;
[A, b, x_true, ProbInfo] = PRtomo(n); % x_true is exact solution A\b

imagesc(reshape(x_true,ProbInfo.xSize))
imagesc(reshape(b,ProbInfo.bSize))

[bn, NoiseInfo] = PRnoise(b, 0.03);
noise = NoiseInfo.noise;
rel_noise = norm(noise)/norm(b);  %0.03
tol  = 1e-6;
maxit = 100;

lambda=100;

%% B approx to A'

tau=0.01;
T = tau* max(A(:));        % threshold = max over all entries of A
B = A.';               % swap indices: B(i,j) = A(j,i)
B(B < T) = 0; 


%% Tomography example
% GMRES
Agmres = A'*A;
bgmres = A'*b;
[x_iterates, error_norm, residual_norm, niters] = gmres_own(B*A, B*bn, x_true, tol, maxit);

figure(1);
semilogy(1:niters, residual_norm, 'o-',1:niters, error_norm,'s-');
xlabel('Iteration k');
ylabel('Relative norm');
legend('||b_{noise} - A x_k||/||b_{noise}||','||x_k - x_{true}||/||x_{true}||','Location','best');
title('GMRES Convergence');
grid on;


%% AB-GMRES

[x_iterates, error_norm, residual_norm, niters] = ABgmres_own(A, B, bn, x_true, tol, maxit);


figure(2);
semilogy(1:niters, residual_norm, 'o-', 1:niters, error_norm, 's-');
xlabel('Iteration k');
ylabel('Relative norm');
legend('||b_{noise} - A x_k||/||b_{noise}||', '||x_k - x_{true}||/||x_{true}||','Location','best');
title('AB-GMRES Convergence');
grid on;

%% BA-GMRES


[x_iterates, error_norm, residual_norm, niters] = BAgmres_own(A, B, bn, x_true, tol, maxit);


figure(3);
semilogy(1:niters, residual_norm, 'o-', 1:niters, error_norm, 's-');
xlabel('Iteration k');
ylabel('Relative norm');
legend('||b_{noise} - A x_k||/||b_{noise}||', '||x_k - x_{true}||/||x_{true}||','Location','best');
title('BA-GMRES Convergence');
grid on;



%% GMRES-Tikhonov


[x_iterates, error_norm, residual_norm, niters] = gmres_tikhonov(B*A, B*bn, x_true, tol, maxit, lambda);
figure(4);
semilogy(1:niters, residual_norm, 'o-', 1:niters, error_norm, 's-');
xlabel('Iteration k');
ylabel('Relative norm');
legend('||b_{noise} - A x_k||/||b_{noise}||', '||x_k - x_{true}||/||x_{true}||','Location','northeast');
title({['GMRES Tikhonov Convergence with B \approx A^T, noise level=' num2str(rel_noise)], ['\tau = '    num2str(tau)], ['\lambda = ' num2str(lambda)] },'Interpreter','tex');
grid on;



%% AB-GMRES Tikhonov


[x_iterates, error_norm, residual_norm, niters] = ABgmres_hybrid(A, B, bn, x_true, tol, maxit, lambda);

figure(5);
semilogy(1:niters, residual_norm, 'o-', 1:niters, error_norm, 's-');
xlabel('Iteration k');
ylabel('Relative norm');
legend('||b_{noise} - A x_k||/||b_{noise}||', '||x_k - x_{true}||/||x_{true}||','Location','northeast');
title({['AB-GMRES Tikhonov Convergence with B \approx A^T, noise level=' num2str(rel_noise)], ['\tau = '    num2str(tau)], ['\lambda = ' num2str(lambda)] },'Interpreter','tex');
grid on; 

%% BA-GMRES Tikhonov


[x_iterates, error_norm, residual_norm, niters] = BAgmres_hybrid(A, B, bn, x_true, tol, maxit, lambda);

figure(6);
semilogy(1:niters, residual_norm, 'o-', 1:niters, error_norm, 's-');
xlabel('Iteration k');
ylabel('Relative norm');
legend('||b_{noise} - A x_k||/||b_{noise}||', '||x_k - x_{true}||/||x_{true}||','Location','northeast');
title({['BA-GMRES Tikhonov Convergence with B \approx A^T with noise level=' num2str(rel_noise)], ['\tau = '    num2str(tau)], ['\lambda = ' num2str(lambda)] },'Interpreter','tex');
grid on;

%%
tau_list = [0.01, 0.1, 0.3, 0.5];
subplot(2,2,1); hold on; grid on
for i=1:length(tau_list)
    tau=tau_list(i);
    T = tau* max(A(:));        % threshold = max over all entries of A
    B = A.';               % swap indices: B(i,j) = A(j,i)
    B(B < T) = 0;

    [~, errAB, ~, ~] = ABgmres_own(A, B, bn, x_true, tol, maxit);
    plot(errAB);
    leg{i} = sprintf('\\tau = %.2f', tau);
end
title('AB-GMRES', 'Interpreter','tex')
xlabel('Iteration k'); ylabel('Relative error')
legend(leg, 'Interpreter','tex', 'Location','northeast')
hold off

subplot(2,2,2)
hold on; grid on
for i=1:length(tau_list)
    tau=tau_list(i);
    T = tau* max(A(:));        % threshold = max over all entries of A
    B = A.';               % swap indices: B(i,j) = A(j,i)
    B(B < T) = 0;

    [~, errBA, ~, ~] = BAgmres_own(A, B, bn, x_true, tol, maxit);
    plot(errBA);
    leg{i} = sprintf('\\tau = %.2f', tau);
end
title('BA-GMRES','Interpreter','tex')
xlabel('Iteration k'); ylabel('Relative error')
legend(leg, 'Interpreter','tex', 'Location','northeast')
hold off

subplot(2,2,3)
hold on; grid on
for i=1:length(tau_list)
    tau=tau_list(i);
    T = tau* max(A(:));        % threshold = max over all entries of A
    B = A.';               % swap indices: B(i,j) = A(j,i)
    B(B < T) = 0;

    [~, errAB, ~, ~] = ABgmres_hybrid(A, B, bn, x_true, tol, maxit,lambda);
    plot(errAB);
    leg{i} = sprintf('\\tau = %.2f', tau);
end
title('Hybrid AB-GMRES', 'Interpreter','tex')
xlabel('Iteration k'); ylabel('Relative error')
legend(leg, 'Interpreter','tex', 'Location','northeast')
hold off

subplot(2,2,4)
hold on; grid on
for i=1:length(tau_list)
    tau=tau_list(i);
    T = tau* max(A(:));        % threshold = max over all entries of A
    B = A.';               % swap indices: B(i,j) = A(j,i)
    B(B < T) = 0;

    [~, errBA, ~, ~] = BAgmres_hybrid(A, B, bn, x_true, tol, maxit,lambda);
    plot(errBA);
    leg{i} = sprintf('\\tau = %.2f', tau);
end
title('Hybrid BA-GMRES', 'Interpreter','tex')
xlabel('Iteration k'); ylabel('Relative error')
legend(leg, 'Interpreter','tex', 'Location','northeast')
hold off
