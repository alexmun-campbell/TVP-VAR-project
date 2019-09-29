%====================================4.1 Main===================================
clear;
clc;

% Load Korobilis (2008) quarterly data
load ydata.dat;
load yearlab.dat;

% We don't standardise the data

%%
%----------------------------------BASICS----------------------------------


Y=ydata;
t=size(Y,1);        % t - The total number of periods in the Y timeseries 
M=size(Y,2);        % M - The dimensionality of Y (i.e. the number of variables)
tau = 40;           % tau - the size of the training sample (the first forty quarters)
p = 2;              % p - number of lags in the VAR model 

%% Generate the Z_t matrix, i.e. the regressors in the model. 

ylag = mlag2(Y,p); % This function generates a 215x6 matrix with p lags of variable Y. 
ylag = ylag(p+tau+1:t,:);  % Then remove our training sample, so now a 173x6 matrix. 

K = M + p*(M^2);            % K is the number of elements in the state vector

% And now we spread out the lagged y data into the Z matrix so it is
% conformable with a beta_t matrix of coefficients

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
t=size(y,2); % t now equals 173

%%
%----------------------MODEL AND GIBBS PRELIMINARIES-----------------------

nrep = 5000;    % Number of replications
nburn = 2000;   % Number of burn-in-draws
it_print = 100; % Print in the screen every "it_print"-th iteration

% We use the first 40 observations (tau) and run a standard OLS (through 
% the function ts_prior to get an estimate for priors for B_0 and Var(B_0). 

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

Btdraw = zeros(K,t);
Bt_postmean = zeros(K,t);
Qmean = zeros(K,K);
Sigmamean = zeros(M,M);

%%
%-------------------------IRF-PRELIMINARIES--------------------------------
nhor = 21;     % The number of periods in the impulse response function. 
shock = diag([zeros(1,M-1) .25]'); % NOT SURE WHAT THIS IS ABOUT?
imp75 = zeros(nrep,M,nhor); 
imp81 = zeros(nrep,M,nhor);
imp96 = zeros(nrep,M,nhor);
bigj = zeros(M,M*p); 
bigj(1:M,1:M) = eye(M);


%% ================ START GIBBS SAMPLING ==================================

tic; % This is just a timer
disp('Number of iterations');

for irep = 1:nrep + nburn    % GIBBS iterations starts here
    % Print iterations - this just updates on the progress of the sampling
    if mod(irep,it_print) == 0
        disp(irep);toc;
    end
    
%% Draw 1: B_t from p(B_t|y,Sigma)
    
    % We use the function 'carter_kohn_hom' to produce a single draw from 
    % the marginal density of B_t conditional on y and sigma
    % i.e. p(B_t|y,Sigma). This results in a 21x173 matrix, corresponding
    % to a single draw of each of the coefficients at each time period. The
    % inputs Sigmadraw and Qdraw are used recursively, so each iteration of
    % the Gibbs sample uses values from the previous iteration. 
    
    [Btdraw] = carter_kohn_hom(y,Z,Sigmadraw,Qdraw,K,M,t,B_0_prmean,B_0_prvar);
    
  
%% Draw 2: Q from p(Q^{-1}|y,B_t) which is i-Wishart

    % We use draw from Q using an Inverse Wishart distribution. The parameters 
    % of the distribution are derrived as equation 23 in the main report.
    % The mean is taken as the inverse of the accumulated sum of squared 
    % errors added to the prior mean, and the variance is simply T.  
    
    % Differencing Btdraw to create the sum of squared errors
    Btemp = Btdraw(:,2:t)' - Btdraw(:,1:t-1)'; 
    sse_2Q = zeros(K,K);
    for i = 1:t-1
        sse_2Q = sse_2Q + Btemp(i,:)'*Btemp(i,:);
    end

    Qinv = inv(sse_2Q + Q_prmean); % find mean to use for Wishart draw
    Qinvdraw = wish(Qinv,t+Q_prvar); % draw from the wishart distribution
    Qdraw = inv(Qinvdraw); % find non-inverse q 
    Qchol = chol(Qdraw); % choelsky decomposition (for IRF analysis)
    
%% Draw 3: Sigma from p(Sigma|y,B_t) which is i-Wishart

    % We draw Sigma using an Inverse Wishart distribution. The parameters 
    % of the distirbution are derrived as equation 21 in the main report. 
    % The mean is taken as the inverse of the sum of squared residuals
    % added to thr prior mean. The variance is simply T. 
    
    % Find residuals using data and the current draw of coefficients
    resids = zeros(M,t);
    for i = 1:t
        resids(:,i) = y(:,i) - Z((i-1)*M+1:i*M,:)*Btdraw(:,i);
    end
    
    % This creates a matrix of the accumulated sum of squared residuals, to
    % be used as the mean parameter in the i-wishart draw below 
    sse_2S = zeros(M,M);
    for i = 1:t
        sse_2S = sse_2S + resids(:,i)*resids(:,i)';
    end
    
    Sigmainv = inv(sse_2S + Sigma_prmean); % find mean to use for the Wishart
    Sigmainvdraw = wish(Sigmainv,t+Sigma_prvar); % draw from the Wishsart distribution
    Sigmadraw = inv(Sigmainvdraw); % turn into non-inverse Sigma
    Sigmachol = chol(Sigmadraw); % Cholesky decomposition 
    
%% IRF 
    % we only apply IRF analysis once we have iterated Gibbs sampling a
    % sufficient number of times
    if irep > nburn; 
        
        % For memory efficiency, we just save the means of the draws. This
        % is achieved by adding the three draw matricies worked out above to 
        % a matricies of accumulated draws from all previous iterations of
        % the gibbs loop (it works because the expected value of the mean
        % is zero. The means for Q and Sigma are then fed back as an input
        % into the draws for B_t (carter_kohn function). 
        
        Bt_postmean = Bt_postmean + Btdraw; 
        Qmean = Qmean + Qdraw;
        Sigmamean = Sigmamean + Sigmadraw;
         
            %% biga is a 6x6 matrix of coefficients of the MA version of 
            % the VAR model. Is is constructed with a 3x3 I matrix in 
            % bottom left corner in order to make it conformable with this
            % VAR(2) model. We are only concerned with non-intercept
            % coefficients (18 of them ) spread across top half.
            % This corresponds to equation 25 in the report. The matrix
            % biga changes in every period of the analysis, because the
            % coefficients are time varying, so we apply the analysis below
            % in every time period. 
            
            biga = zeros(M*p,M*p); %6x6
            for j = 1:p-1
                biga(j*M+1:M*(j+1),M*(j-1)+1:j*M) = eye(M); % insert eye to bottom left corner
            end

            for i = 1:t 
                bbtemp = Btdraw(M+1:K,i);  % get the draw of B(t) at time i=1,...,T  (exclude intercept)
                splace = 0;
                for ii = 1:p
                    for iii = 1:M
                        biga(iii,(ii-1)*M+1:ii*M) = bbtemp(splace+1:splace+M,1)'; %load non-intercept terms
                        splace = splace + M;
                    end
                end
                
                %% create the shock matrix 
                % st dev matrix for structural VAR
                shock = Sigmachol';   % First shock is the Cholesky of the VAR covariance
                diagonal = diag(diag(shock)); 
                shock = inv(diagonal)*shock;    % Unit initial shock 
                
                % Now get impulse responses for 1 through nhor future
                % periods. impresp is a 3x63 matrix which contains 9
                % response values in total for each period, 3 for each 
                % variable. These three responses correspond to the three
                % possible shocks that are impelemented in the schock
                % matrix, which contains 3 sets of shocks. 
                % bigai is updated through mulitiplication with the 
                % coefficient matrix after each time period. 
                
                impresp = zeros(M,M*nhor); % matrix to store initial response at each period
                impresp(1:M,1:M) = shock;  % First shock is the Cholesky of the VAR covariance
                bigai = biga;
                for j = 1:nhor-1
                    impresp(:,j*M+1:(j+1)*M) = bigj*bigai*bigj'*shock; % compute the value of Y for each IRF time period 
                    bigai = bigai*biga; % update the coefficient matrix for next period
                end

                % The secion below filters the responses that we have
                % calculated. We keep only the responses that use beta
                % coefficients from the periods 1975.1, 1981.3, and 1996.1.
                % We also filter out responses to all but the third set of
                % innovations included in the schock vectore - i.e. the
                % unit shock to the third variable. 
                
                if yearlab(i,1) == 1975.00;   % 1975:Q1
                    impf_m = zeros(M,nhor);
                    jj=0;
                    for ij = 1:nhor
                        jj = jj + M;    % restrict to the M-th equation, the interest rate
                        impf_m(:,ij) = impresp(:,jj);
                    end
                    imp75(irep-nburn,:,:) = impf_m; % store draws of responses
                end
                if yearlab(i,1) == 1981.50;   % 1981:Q3
                    impf_m = zeros(M,nhor);
                    jj=0;
                    for ij = 1:nhor
                        jj = jj + M;    % restrict to the M-th equation, the interest rate
                        impf_m(:,ij) = impresp(:,jj);
                    end
                    imp81(irep-nburn,:,:) = impf_m;  % store draws of responses
                end
                if yearlab(i,1) == 1996.00;   % 1996:Q1
                    impf_m = zeros(M,nhor);
                    jj=0;
                    for ij = 1:nhor
                        jj = jj + M;    % restrict to the M-th equation, the interest rate
                        impf_m(:,ij) = impresp(:,jj);
                    end
                    imp96(irep-nburn,:,:) = impf_m;  % store draws of responses
                end
            end %END geting impulses for each time period 
        end %END the impulse response calculation section   
    end % END saving after burn-in results 
end %END main Gibbs loop (for irep = 1:nrep+nburn)
clc;
toc; % Stop timer and print total time
%% ================ END GIBBS SAMPLING ==================================



