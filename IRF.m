%====================================IRF===================================
clear;
clc;

% Load Korobilis (2008) quarterly data
load ydata.dat;
load yearlab.dat;

%----------------------------------BASICS----------------------------------
Y=ydata;
t=size(Y,1);        % t is the time-series observations of Y
M=size(Y,2);        % M is the dimensionality of Y (i.e. the number of variables)
tau = 40;           % tau is the size of the training sample (the first forty quarters)
p = 2;              % p is number of lags in the VAR part

% Generate lagged Y matrix which will be used in the X matrix
ylag = mlag2(Y,p);  % Function that generates a matrix vid p lags of variable Y
% Form RHS matrix X_t = [1 y_t-1 y_t-2 ... y_t-k] for t=1:T
ylag = ylag(p+tau+1:t,:);   % Removing our training sample and two lags from Y

K = M + p*(M^2);            % K is the number of elements in the state vector

% Create Z_t matrix, matrix for our time varying coefficients
Z = zeros((t-tau-p)*M,K);   
for i = 1:t-tau-p
    ztemp = eye(M);
    for j = 1:p        
        xtemp = ylag(i,(j-1)*M+1:j*M);
        xtemp = kron(eye(M),xtemp);
        ztemp = [ztemp xtemp];
    end
    Z((i-1)*M+1:i*M,:) = ztemp;
end

% Redefine our variables to exclude the training sample, taking t from 215
% to 173
y = Y(tau+p+1:t,:)';
yearlab = yearlab(tau+p+1:t);
t=size(y,2);

%-----------------------------PRELIMINARIES--------------------------------
nrep = 5000;    % Number of replications
nburn = 2000;   % Number of burn-in-draws
it_print = 100; % Print in the screen every "it_print"-th iteration

% We use the first 40 observations (tau) and run a standard OLS to get an
% estimate for priors for B_0 and Var(B_0)

[B_OLS,VB_OLS,A_OLS,sigma_OLS,VA_OLS]= ts_prior(Y,tau,M,p);

% Given the distributions we have, we now have to define our priors for B,
% Q and Sigma. These are set in accordance with how they are set in the
% paper. 

B_0_prmean = B_OLS;
B_0_prvar = 4*VB_OLS; 

Q_prmean = ((0.01).^2)*tau*VB_OLS;
Q_prvar = tau;

Sigma_prmean = eye(M);
Sigma_prvar = M+1;

% To start the Kalman filtering however, we need to assign specific values
% to Q and Sigma in order to be able to start the sampling. The only
% restriction on these values are that they have to be in the support of
% their respective distributions. We select our priors in the same way as
% in the paper. 

consQ = 0.0001;
Qdraw = consQ*eye(K);
Sigmadraw = 0.1*eye(M); %same as above, just needs to be in the support
Qchol = sqrt(consQ)*eye(K);

% Now we create some matrices for storage that will be filled in once we
% start the Gibbs sampling.

Btdraw = zeros(K,t)
Bt_postmean = zeros(K,t);
Qmean = zeros(K,K);
Sigmamean = zeros(M,M);

%-------------------------IRF-PRELIMINARIES--------------------------------
