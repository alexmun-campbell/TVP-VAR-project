%=============================HIERARCHICAL=================================
clear;
clc;

% Loading our data, this below generates some random numbers?
[Y, beta_true,theta_true] = hierch_tvp_dgp();

%----------------------------------BASICS----------------------------------

% Basics are identical to what we do in the first TVP-VAR model

T=size(Y,1);  % T is the number of observations in Y
M=size(Y,2);  % M is the dimensionality of Y (i.e. the number of variables)
p = 1;        % p is number of lags in the VAR part

% Generate lagged Y matrix which will be used in the X matrix
ylag = mlag2(Y,p);      % Function that generates a matrix vid p lags of variable Y
% Form RHS matrix X_t = [1 y_t-1 y_t-2 ... y_t-k] for t=1:T
ylag = ylag(p+1:T,:);   % Remove the first time period

K = p*(M^2);  % K is the number of elements in the state vector
Z = zeros((T-p)*M,K);
for i = 1:T-p
    ztemp = []; %eye(M) - This is the only difference, in first, it is eye(M)
    for j = 1:p        
        xtemp = ylag(i,(j-1)*M+1:j*M);
        xtemp = kron(eye(M),xtemp);
        ztemp = [ztemp xtemp]; 
    end
    Z((i-1)*M+1:i*M,:) = ztemp;
end

y = Y(p+1:T,:)';
T=size(y,2);

%-----------------------------PRELIMINARIES--------------------------------
nrep = 3000;  % Number of replications
nburn = 3000;   % Number of burn-in-draws
it_print = 100;  %Print in the screen every "it_print"-th iteration

% We create our priors for the variables, not using an informed prior for
% theta this time around.

theta_0_prmean = zeros(K,1);
theta_0_prvar = 4*eye(K);

Q_prmean = eye(K);
Q_prvar = K+1;

R_prmean = eye(K);
R_prvar = K+1;

Sigma_prmean = eye(M);
Sigma_prvar = M+1;

% Setting initializing values for Q, R, Sigma and theta.

Q = 0.0001*eye(K);
Qinv = inv(Q);
R = Q;
Rinv = inv(R);
Sigma = 0.1*eye(M);
Sigmainv = inv(Sigma);
theta_t = theta_true;

% Creating some matrices for storage of temporary and posterior data

beta_t = zeros(T,K);
A_0 = eye(K); % We are assuming A_0 is the identity matrix in this version       
beta_t_mean = zeros(T,K);
theta_t_mean = zeros(T,K);
Q_mean = zeros(K,K);
R_mean = zeros(K,K);
Sigma_mean = zeros(M,M);
A_0_mean = zeros(K,K);

%%
%-----------------------------GIBBS SAMPLING-------------------------------
tic; % This is just a timer
disp('Number of iterations');

for irep = 1:nrep + nburn
    if mod(irep,it_print) == 0
        disp(irep);toc;
    end
    %----------------------------------------------------------------------
    % Step 1: Sample beta_t ~ N(beta_post_mean,beta_post_var)
    %----------------------------------------------------------------------
    for i=1:T
        beta_post_var = inv(Qinv + Z((i-1)*M+1:i*M,:)'*Sigmainv*Z((i-1)*M+1:i*M,:));
        beta_post_mean = beta_post_var*(Qinv*(A_0*theta_t(i,:)') + Z((i-1)*M+1:i*M,:)'*Sigmainv*y(:,i));
        beta_t(i,:) = beta_post_mean + chol(beta_post_var)'*randn(K,1);    
    end
    %----------------------------------------------------------------------
    % Step 2: Sample Q ~ iW(v_q,S_q)
    %----------------------------------------------------------------------
    sse_q = 0;
    for i=1:T
        sse_q =  sse_q + (beta_t(i,:) - theta_t(i,:)*A_0)'*(beta_t(i,:) - theta_t(i,:)*A_0);
    end
    v_q = T + Q_prvar;
    S_q = inv(Q_prmean + sse_q);
    Qinv = wish(S_q,v_q);
    Q = inv(Qinv);
    %----------------------------------------------------------------------
    % Step 3: Sample theta_t using Carter and kohn
    %----------------------------------------------------------------------    
    [theta_tc,log_lik] = carter_kohn_hom2(beta_t',A_0,Q,R,K,K,T,theta_0_prmean,theta_0_prvar);
    theta_t = theta_tc';
    %----------------------------------------------------------------------
    % Step 4: Sample R ~ iW(v_r,S_r)
    %----------------------------------------------------------------------
    sse_r = 0;
    theta_temp = theta_t(2:T,:) - theta_t(1:T-1,:);
    for i=1:T-1
        sse_r =  sse_r + (theta_temp(i,:))'*(theta_temp(i,:));
    end
    v_r = T + R_prvar;
    S_r = inv(R_prmean + sse_r);
    Rinv = wish(S_r,v_r);
    R = inv(Rinv);
    % ---------------------------------------------------------------------
    %   Step 5: Sample Sigma from M(Sigma|y,B_t) which is i-Wishart
    % ---------------------------------------------------------------------
    yhat = zeros(M,T);
    for i = 1:T
        yhat(:,i) = y(:,i) - Z((i-1)*M+1:i*M,:)*beta_t(i,:)';
    end
    
    sse_S = zeros(M,M);
    for i = 1:T
        sse_S = sse_S + yhat(:,i)*yhat(:,i)';
    end
    
    Sinv = inv(sse_S + Sigma_prmean);
    Sigmainv = wish(Sinv, T + Sigma_prvar);
    Sigma = inv(Sigmainv);
    Sigmachol = chol(Sigma);
    
    %-----------------------SAVE AFTER-BURN-IN DRAWS-----------------------
    if irep > nburn;
         beta_t_mean = beta_t_mean + beta_t;
         theta_t_mean = theta_t_mean + theta_t;
         Q_mean = Q_mean + Q;
         R_mean = R_mean + R;
         Sigma_mean = Sigma_mean + Sigma;
         A_0_mean = A_0_mean + A_0;
    end % END saving after burn-in results
end %END main Gibbs loop (for irep = 1:nrep+nburn)
clc;
toc; % Stop timer and print total time

%-------------------------END OF GIBBS SAMPLER-----------------------------

% Calculate our moments 

beta_t_mean = beta_t_mean./nrep;        
theta_t_mean = theta_t_mean./nrep;
Q_mean = Q_mean./nrep;        
R_mean = R_mean./nrep;         
Sigma_mean = Sigma_mean./nrep;
A_0_mean = A_0_mean./nrep;