function plot_filter_factors()

%   Generate shaw(32) test problem, run 4 GMRES variants with bounds,
%   compute theoretical vs empirical filter factors, and plot them.

  %% 1) Set up Shaw test problem
  n = 32;
  [A, b, x_true] = shaw(n);
  %[A, b, x_true] = heat(n);
  %[A, b, x_true] = deriv2(n);
  %[A, b, x_true, ProbInfo] = PRtomo(n);


  %% 2) Algorithm parameters
  tol     = 1e-6;
  maxit   = n;          % full dimension
  lambda  = 1e-3;
  B       = A';         % matched preconditioner
  DeltaM  = 1e-5*randn(n);

  %% 3) Run each method & collect phi,dPhi
  %  (non-hybrid AB-GMRES)
  [x_ab,~,~,it_ab, phi_ab, ~] = ...
    ABgmres_nonhybrid_bounds(A, B, b, x_true, tol, maxit, DeltaM);
  %  (non-hybrid BA-GMRES)
  [x_ba,~,~,it_ba, phi_ba, ~] = ...
    BAgmres_nonhybrid_bounds(A, B, b, x_true, tol, maxit, DeltaM);
  %  (hybrid AB-GMRES)
  [x_hab,~,~,it_hab, phi_hab, ~] = ...
    ABgmres_hybrid_bounds(A, B, b, x_true, tol, maxit, lambda, DeltaM);
  %  (hybrid BA-GMRES)
  [x_hba,~,~,it_hba, phi_hba, ~] = ...
    BAgmres_hybrid_bounds(A, B, b, x_true, tol, maxit, lambda, DeltaM);

  %% 4) Compute empirical filters at final iterates
  [U,S,V] = svd(A,'econ');
  sigma   = diag(S);
  d       = U' * b;

  Phi_emp_ab  = sigma .* (V' * x_ab) ./ d;
  Phi_emp_ba  = sigma .* (V' * x_ba) ./ d;
  Phi_emp_hab = sigma .* (V' * x_hab) ./ d;
  Phi_emp_hba = sigma .* (V' * x_hba) ./ d;

  kmax = length(sigma);

  %% 5) Plot everything
 % Determine how many modes we can plot without running off the end
k_theo = [length(phi_ab), length(phi_ba), length(phi_hab), length(phi_hba)];
k_emp  = [length(Phi_emp_ab), length(Phi_emp_ba), length(Phi_emp_hab), length(Phi_emp_hba)];
kmin   = min([k_theo, k_emp]);

modes = 1:kmin;

figure; hold on;
  lw = 1.8;

  % Theoretical
  plot(modes, phi_ab(1:kmin),  '--','LineWidth',lw,'Color',[.2 .6 .2]);
  plot(modes, phi_ba(1:kmin),  '-.','LineWidth',lw,'Color',[.2 .2 .6]);
  plot(modes, phi_hab(1:kmin), '-','LineWidth',lw,'Color',[.8 .3 .3]);
  plot(modes, phi_hba(1:kmin), ':','LineWidth',lw,'Color',[.6 .2 .6]);

  % Empirical
  plot(modes, Phi_emp_ab(1:kmin),  '-x','MarkerSize',6,'Color',[.2 .6 .2]);
  plot(modes, Phi_emp_ba(1:kmin),  '-s','MarkerSize',6,'Color',[.2 .2 .6]);
  plot(modes, Phi_emp_hab(1:kmin), '-d','MarkerSize',6,'Color',[.8 .3 .3]);
  plot(modes, Phi_emp_hba(1:kmin), '-o','MarkerSize',6,'Color',[.6 .2 .6]);
hold off;

xlabel('Mode index \it{i}');
ylabel('Filter factor / empirical filter');
title('Theoretical vs Empirical Filter Factors');
legend({ ...
  'AB (theory)','BA (theory)','hAB (theory)','hBA (theory)', ...
  'AB (emp)','BA (emp)','hAB (emp)','hBA (emp)'}, ...
  'Location','Best');
grid on;
%% EMPIRICAL VS THEORICAL FILTER FACTORS SUBPLOTS
% 5) Four subplots, theory vs empirical, guarded against length/complexity
figure('Position',[200 200 800 600]);

% 1) non‑hybrid AB
kmin_ab = min(numel(phi_ab),  numel(Phi_emp_ab));
i_ab    = 1:kmin_ab;
subplot(2,2,1);
plot(i_ab, real(phi_ab(1:kmin_ab)),  '--','LineWidth',1.6); hold on;
plot(i_ab, real(Phi_emp_ab(1:kmin_ab)),'x-','MarkerSize',6);
hold off;
xlabel('i'); ylabel('\phi_i');
title('AB‑GMRES (non‑hybrid)');
legend('theoretical','empirical','Location','Best');
grid on;

% 2) non‑hybrid BA
kmin_ba = min(numel(phi_ba),  numel(Phi_emp_ba));
i_ba    = 1:kmin_ba;
subplot(2,2,2);
plot(i_ba, real(phi_ba(1:kmin_ba)),  '-.','LineWidth',1.6); hold on;
plot(i_ba, real(Phi_emp_ba(1:kmin_ba)),'s-','MarkerSize',6);
hold off;
xlabel('i'); ylabel('\phi_i');
title('BA‑GMRES (non‑hybrid)');
legend('theoretical','empirical','Location','Best');
grid on;

% 3) hybrid AB
kmin_hab = min(numel(phi_hab),  numel(Phi_emp_hab));
i_hab    = 1:kmin_hab;
subplot(2,2,3);
plot(i_hab, real(phi_hab(1:kmin_hab)), '-','LineWidth',1.6); hold on;
plot(i_hab, real(Phi_emp_hab(1:kmin_hab)),'d-','MarkerSize',6);
hold off;
xlabel('i'); ylabel('\phi_i');
title('AB‑GMRES (hybrid)');
legend('theoretical','empirical','Location','Best');
grid on;

% 4) hybrid BA
kmin_hba = min(numel(phi_hba), numel(Phi_emp_hba));
i_hba    = 1:kmin_hba;
subplot(2,2,4);
plot(i_hba, real(phi_hba(1:kmin_hba)), ':','LineWidth',1.6); hold on;
plot(i_hba, real(Phi_emp_hba(1:kmin_hba)),'o-','MarkerSize',6);
hold off;
xlabel('i'); ylabel('\phi_i');
title('BA‑GMRES (hybrid)');
legend('theoretical','empirical','Location','Best');
grid on;

end
