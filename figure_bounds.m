%%
n = 64;
[A, b, x_true, ProbInfo] = PRtomo(n);        % A is m×n,  b is m×1,  x_true = A\b
figure; imagesc(reshape(x_true,ProbInfo.xSize)); 
title('Exact solution x\_true');
axis image off; colorbar;

figure; imagesc(reshape(b,ProbInfo.bSize));
title('Noiseless data b');
axis image off; colorbar;

%-----------------------------%
% 2) Add 3% Gaussian noise
%-----------------------------%
[bn, NoiseInfo] = PRnoise(b, 0.03);
b_noisy    = bn;
noise      = NoiseInfo.noise;
rel_noise  = norm(noise)/norm(b);  % ≈0.03
figure; imagesc(reshape(b_noisy,ProbInfo.bSize));
title(sprintf('Noisy data (%.2f%% noise)',rel_noise*100));
axis image off; colorbar;

%-----------------------------%
% 3) Set parameters & operators
%-----------------------------%
tol   = 1e-6;
maxit = 100;
lambda= 50;

% For AB‐GMRES, use matched back-projector
B = A';

% Build a small random perturbation ΔM (same size as M = A*B)
m      = size(A,1);
epsilon= 1e-3;                   % choose your perturbation magnitude
DeltaM_m = epsilon*randn(m,m);

n      = size(A,2);          % number of columns of A = size of x              
DeltaM_n = epsilon * randn(n,n);



%% Filter‐factor comparisons for hybrid AB‐GMRES and BA‐GMRES


% 1) Compute for BA‐GMRES
tic;
[x_ba, err_ba, res_ba, it_ba, phi_ba, dPhi_ba] = BAgmres_hybrid_bounds(A, B, b, x_true, tol, maxit, lambda, DeltaM_n);
time_ba = toc;
fprintf('BA-GMRES took %.3f seconds and %d iterations\n', time_ba, it_ba);
phi_ba_pert = phi_ba + dPhi_ba;


% 2) Compute for AB‐GMRES
tic;
[x_ab, err_ab, res_ab, it_ab, phi_ab, dPhi_ab] = ABgmres_hybrid_bounds(A, B, b, x_true, tol, maxit, lambda, DeltaM_m);
time_ab = toc;
fprintf('AB-GMRES took %.3f seconds and %d iterations\n', time_ab, it_ab);
phi_ab_pert = phi_ab + dPhi_ab;


%% Plot Hyrbid BA-GMRES and BA‐GMRES filter factors

figure;

subplot(2,1,1)
plot(1:length(phi_ba), phi_ba, 'o-', 'LineWidth',1.5, 'DisplayName','\phi_{BA}');
hold on;
plot(1:length(phi_ba_pert), phi_ba_pert, 'x--', 'LineWidth',1.5, 'DisplayName','\phi_{BA} + \Delta\phi_{BA}');
xlabel('Index i');
ylabel('Filter factor value');
title('BA‐GMRES: Unperturbed vs. First‐Order Perturbed Filters');
legend('Location','best');
grid on;

subplot(2,1,2)
plot(1:length(phi_ab), phi_ab, 'o-', 'LineWidth',1.5, 'DisplayName','\phi_{AB}');
hold on;
plot(1:length(phi_ab_pert), phi_ab_pert, 'x--', 'LineWidth',1.5, 'DisplayName','\phi_{AB} + \Delta\phi_{AB}');
xlabel('Index i');
ylabel('Filter factor value');
title('AB‐GMRES: Unperturbed vs. First‐Order Perturbed Filters');
legend('Location','best');
grid on;




%% Plot Hyrbid BA-GMRES and BA‐GMRES filter factors (semilog-scale)

figure;

subplot(2,1,1)
semilogy(1:length(phi_ba), phi_ba, 'o-', 'LineWidth',1.5, 'DisplayName','\phi_{BA}');
hold on;
semilogy(1:length(phi_ba_pert), phi_ba_pert, 'x--', 'LineWidth',1.5, 'DisplayName','\phi_{BA} + \Delta\phi_{BA}');
xlabel('Index i');
ylabel('Filter factor value');
title('BA‐GMRES: Unperturbed vs. First‐Order Perturbed Filters');
legend('Location','best');
grid on;

subplot(2,1,2)
semilogy(1:length(phi_ab), phi_ab, 'o-', 'LineWidth',1.5, 'DisplayName','\phi_{AB}');
hold on;
semilogy(1:length(phi_ab_pert), phi_ab_pert, 'x--', 'LineWidth',1.5, 'DisplayName','\phi_{AB} + \Delta\phi_{AB}');
xlabel('Index i');
ylabel('Filter factor value');
title('AB‐GMRES: Unperturbed vs. First‐Order Perturbed Filters');
legend('Location','best');
grid on;

%%

i_ba = 1:it_ba;
i_ab = 1:it_ab;
figure;

% BA-GMRES unperturbed & perturbed filters
subplot(2,2,1);
plot(i_ba, phi_ba, '.-','LineWidth',1.2,'MarkerSize',6); hold on;
plot(i_ba, phi_ba_pert, 'o--','LineWidth',1.2,'MarkerSize',3);
hold off;
xlabel('Filter index i');
ylabel('\phi_{ba,i}');
title('BA-GMRES: \phi_{ba} and \phi_{ba}+\delta\phi_{ba}');
legend('unperturbed','perturbed','Location','Best');
grid on;

% AB-GMRES unperturbed & perturbed filters
subplot(2,2,2);
plot(i_ab, phi_ab, '.-','LineWidth',1.2,'MarkerSize',6); hold on;
plot(i_ab, phi_ab_pert, 'o--','LineWidth',1.2,'MarkerSize',3);
hold off;
xlabel('Filter index i');
ylabel('\phi_{ab,i}');
title('AB-GMRES: \phi_{ab} and \phi_{ab}+\delta\phi_{ab}');
legend('unperturbed','perturbed','Location','Best');
grid on;

% BA-GMRES perturbation magnitude
subplot(2,2,3);
semilogy(i_ba, (dPhi_ba), '.-','LineWidth',1.2,'MarkerSize',8);
xlabel('Filter index i');
ylabel('\delta\phi_{ba,i}');
title('BA-GMRES: Perturbation bound');
grid on;

% AB-GMRES perturbation magnitude
subplot(2,2,4);
semilogy(i_ab, (dPhi_ab), '.-','LineWidth',1.2,'MarkerSize',8);
xlabel('Filter index i');
ylabel('\delta\phi_{ab,i}');
title('AB-GMRES: Perturbation bound');
grid on;


%% Filter‐factor comparisons for hybrid AB‐GMRES and BA‐GMRES, with stopping

method = 'GCV';   % or 'DP'


% 1) BA-GMRES
tic;
[x, error_norm, residual_norm, niters, phi, dPhi, lambda_vec] = ...
    BAgmres_hybrid_bounds_stopping(A, B, b_noisy, x_true, tol, maxit, lambda, DeltaM_n, method);        % pick λ at each step by GCV

time_ba_stopping = toc;
fprintf('BA‐GMRES took %.3f s, %d iters, λ_opt = %.3e\n', time_ba_stopping, niters, lambda_vec(niters));

% 2) AB-GMRES
tic;
[x_ab, error_norm_ab, residual_norm_ab, iters_ab, phi_ab, dPhi_ab, lambda_vec_ab] = ...
    ABgmres_hybrid_bounds_stopping(A, B, b_noisy, x_true, tol, maxit, lambda, DeltaM_m, method);      

time_ab_stopping = toc;
fprintf('AB‐GMRES took %.3f s, %d iters, λ_opt = %.3e\n', time_ab_stopping, iters_ab, lambda_vec_ab(end));


%% Plot unperturbed vs. first-order perturbed filters for BA-GMRES with stopping
k = niters;
i = (1:length(phi))';   
lambda_opt_ba = lambda_vec(k);

figure;
plot(i, phi, 'b-', 'LineWidth',1.8); hold on
plot(i, phi + dPhi, 'r--', 'LineWidth',1.5);
hold off

xlabel('Mode index \it i');
ylabel('Filter value');
legend('\phi_i','\phi_i + d\phi_i','Location','Best');
title(sprintf('BA‐GMRES Filters at k=%d (%s \\lambda=%.2e)', k, method, lambda_opt_ba), 'Interpreter','none');
grid on
axis tight


%% Plot unperturbed vs. first-order perturbed filters for AB-GMRES with stopping
k_ab      = iters_ab;
i         = (1:length(phi_ab))';  
lambda_opt_ab = lambda_vec_ab(end);

figure;
plot(i,         phi_ab,        'b-',  'LineWidth',1.8); hold on
plot(i, phi_ab + dPhi_ab,        'r--', 'LineWidth',1.5);
hold off

xlabel('Mode index \it i');
ylabel('Filter value');
legend('\phi_i','\phi_i + d\phi_i','Location','Best');
title( sprintf('AB‐GMRES Filters at k=%d (%s \\lambda=%.2e)', k_ab, method, lambda_opt_ab), ...
       'Interpreter','none' );
grid on
axis tight


























%% Additional plots

% 1) GMRES residual history (semilogy) :contentReference[oaicite:0]{index=0}
figure;
semilogy(1:length(res_ba), res_ba, 's-','LineWidth',1.5, 'DisplayName','BA-residual');
hold on;
semilogy(1:length(res_ab), res_ab, 'o-','LineWidth',1.5, 'DisplayName','AB-residual');
hold off;
xlabel('Iteration k');
ylabel('Relative residual');
title('GMRES Residual History');
legend('Location','best');
grid on;

% 2) Error norm vs. iteration (semiconvergence) :contentReference[oaicite:1]{index=1}
figure;
plot(1:length(err_ba), err_ba, 's-','LineWidth',1.5,'DisplayName','BA-error');
hold on;
plot(1:length(err_ab), err_ab, 'o-','LineWidth',1.5,'DisplayName','AB-error');
hold off;
xlabel('Iteration k');
ylabel('Relative error');
title('Error Norm vs. Iteration');
legend('Location','best');
grid on;

% 3) L-curve for BA-GMRES (solution norm vs. residual norm) :contentReference[oaicite:2]{index=2}
norm_x_ba = err_ba * norm(x_true);  % since err_ba = ||x_k - x_true||/||x_true||
figure;
loglog(res_ba, norm_x_ba, 's-','LineWidth',1.5);
xlabel('Residual norm ||r_k||/||b||');
ylabel('Solution norm ||x_k||');
title('L-curve (BA-GMRES)');
grid on;

% 4) Picard plot (σ_i vs. |u_i^T b|) :contentReference[oaicite:3]{index=3}
[U_A, S_A, V_A] = svd(full(A), 'econ');
sigma_A = diag(S_A);
coeffs = U_A' * b_noisy;
figure;
loglog(sigma_A, abs(coeffs), '.-','LineWidth',1.2);
xlabel('Singular values \sigma_i');
ylabel('|u_i^T b_{noisy}|');
title('Picard Plot');
grid on;

% 5) Backprojection of noisy data :contentReference[oaicite:4]{index=4}
bp_img = reshape(A' * b_noisy, ProbInfo.xSize);
figure; imagesc(bp_img);
axis image off; colorbar;
title('Unfiltered Backprojection of Noisy Data');