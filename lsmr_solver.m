function [x, err_hist, res_hist, ar_hist, iters] = lsmr_solver(A, b, x_true, tol, maxit)

    if nargin < 4 || isempty(tol),   tol   = 1e-6; end
    [m,n] = size(A);
    if nargin < 5 || isempty(maxit), maxit = min(m,n); end

    x = zeros(n,1);

    % GKB
    u = b;
    beta = norm(u);
    if beta > 0, u = u/beta; end

    v = A.'*u;
    alpha = norm(v);
    if alpha > 0, v = v/alpha; end

    %  LSMR scalars  
    zetabar  = alpha*beta;       
    alphabar = alpha;            
    rho      = 1;                
    rhobar   = 1;                
    cbar     = 1; sbar = 0;      

    h    = v;                   
    hbar = zeros(n,1);           

    err_hist = nan(maxit,1);
    res_hist = zeros(maxit,1);
    ar_hist  = zeros(maxit,1);

    for k = 1:maxit
        %  GKB step 
        u = A*v - alpha*u;
        beta = norm(u);
        if beta > 0, u = u/beta; end

        v = A.'*u - beta*v;
        alpha = norm(v);
        if alpha > 0, v = v/alpha; end

        alphahat = alphabar;      
        rhoold = rho;
        rho = hypot(alphahat, beta);
        c = alphahat / rho;
        s = beta      / rho;

        thetanew = s*alpha;
        alphabar = c*alpha;

        rhobarold = rhobar;
        thetabar  = sbar * rho;
        rhobar    = hypot(cbar*rho, thetanew);
        cbar      = (cbar*rho) / rhobar;
        sbar      = thetanew   / rhobar;

   
        zeta    = cbar * zetabar;
        zetabar = -sbar * zetabar;

        if k == 1
            hbar = h;   
        else
            hbar = h - (thetabar * rho)/(rhoold * rhobarold) * hbar;
        end
        x = x + (zeta / (rho * rhobar)) * hbar;
        h = v - (thetanew / rho) * h;

        r = b - A*x;
        res_hist(k) = norm(r) / (norm(b)+eps);
        ar_hist(k)  = norm(A.'*r) / (norm(A,'fro')*max(norm(r),eps));  
        if nargin >= 3 && ~isempty(x_true)
            err_hist(k) = norm(x - x_true) / norm(x_true);
        end

        if res_hist(k) < tol, break; end
    end

    iters    = k;
    err_hist = err_hist(1:iters);
    res_hist = res_hist(1:iters);
    ar_hist  = ar_hist(1:iters);
end


