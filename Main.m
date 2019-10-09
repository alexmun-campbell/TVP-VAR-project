%====================================4.1 Main===================================
clear;
clc;

% Load Korobilis (2008) quarterly data
load ydata.dat; % data 
load yearlab.dat; % data labels 

%%
%----------------------------------BASICS----------------------------------


Y=ydata;
t=size(Y,1);        % t - The total number of periods in the raw data (t=215)
M=size(Y,2);        % M - The dimensionality of Y (i.e. the number of variables)(M=3)
tau = 40;           % tau - the size of the training sample (the first forty quarters)
p = 2;              % p - number of lags in the VAR model 

%% Generate the Z_t matrix, i.e. the regressors in the model. 

ylag = mlag2(Y,p);          % This function generates a 215x6 matrix with p lags of variable Y. 
ylag = ylag(p+tau+1:t,:);   % Then remove our training sample, so now a 173x6 matrix. 

K = M + p*(M^2);            % K is the number of elements in the state vector

% Here we distribute the lagged y data into the Z matrix so it is
% conformable with a beta_t matrix of coefficients. 

%COPY GREY BOX FROM REPORT 

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

% Redefine our variables to exclude the training sample and the first two
% lags that we take as given, taking total number of periods (t) from 215 
% to 173. 

y = Y(tau+p+1:t,:)';
yearlab = yearlab(tau+p+1:t);
t=size(y,2); % t now equals 173

%% ----------------------MODEL AND GIBBS PRELIMINARIES-----------------------

nrep = 5000;    % Number of sample draws 
nburn = 2000;   % Number of burn-in-draws
it_print = 100; % Print in the screen every "it_print"-th iteration


%% INITIAL STATE VECTOR PRIOR  

% We use the first 40 observations (tau) to run a standard OLS of the
% measurement equation, using the function ts_prior. The result is
% estimates for priors for B_0 and Var(B_0). 

[B_OLS,VB_OLS]= ts_prior(Y,tau,M,p);

% Given the distributions we have, we now have to define our priors for B,
% Q and Sigma. These are set in accordance with how they are set in 
% Primiceri (2005). These are the hyperparameters of the beta, Q and sigma
% initial priors. 

B_0_prmean = B_OLS;
B_0_prvar = 4*VB_OLS; 

Q_prmean = ((0.01).^2)*tau*VB_OLS;
Q_prvar = tau;

Sigma_prmean = eye(M);
Sigma_prvar = M+1;

% To start the Kalman filtering asign arbitrary values that are in support
% of their priors, to Q and sigma. 

consQ = 0.0001;
Qdraw = consQ*eye(K);
Sigmadraw = 0.1*eye(M); %same as above, just needs to be in the support

% Now we create some matrices for storage that will be filled in once we
% start the Gibbs sampling.
Btdraw = zeros(K,t); 
Bt_postmean = zeros(K,t);
Qmean = zeros(K,K);
Sigmamean = zeros(M,M);

%% -------------------------IRF-PRELIMINARIES------------------------------
nhor = 21;     % The number of periods in the impulse response function. 

% matricies containing IRFs for 1975q1, 1981q3, 1996q1. The dimensions
% correspond to the iterations of the gibbs sample, each of the variables,
% and each of the 21 periods of the IRF analysis. 

imp75 = zeros(nrep,M,nhor); 
imp81 = zeros(nrep,M,nhor);
imp96 = zeros(nrep,M,nhor);

% This corresponds to equation X in report. 
bigj = zeros(M,M*p); 
bigj(1:M,1:M) = eye(M);


%% ================ START GIBBS SAMPLING ==================================

tic; % This is just a timer
disp('Number of iterations');


for irep = 1:nrep + nburn    % 7000 gibbs iterations starts here
    % Print iterations - this just updates on the progress of the sampling
    if mod(irep,it_print) == 0
        disp(irep);toc;
    end
    
%% Draw 1: B_t from p(B_t|y,Sigma)
    
    % We use the function 'carter_kohn_hom' to to run the FFBS algorithm. 
    % This results in a 21x173 matrix, corresponding 
    % to one Gibbs sample draw of each of the coefficients in each time 
    % period. The
    % inputs Sigmadraw and Qdraw are updated for each Gibbs sample repetition.
    [Btdraw] = carter_kohn_hom(y,Z,Sigmadraw,Qdraw,K,M,t,B_0_prmean,B_0_prvar);
    
  
%% Draw 2: Q from p(Q^{-1}|y,B_t) which is i-Wishart

    % We draw Q from an Inverse Wishart distribution. The parameters 
    % of the distribution are derrived as equation X in the main report.
    % The mean is taken as the inverse of the accumulated sum of squared 
    % errors added to the prior mean, and the variance is simply T.  
    
    % Differencing Btdraw to create the sum of squared errors
    Btemp = Btdraw(:,2:t)' - Btdraw(:,1:t-1)'; 
    sse_2Q = zeros(K,K);
    for i = 1:t-1
        sse_2Q = sse_2Q + Btemp(i,:)'*Btemp(i,:);
    end

    Qinv = inv(sse_2Q + Q_prmean);      % compute mean to use for Wishart draw
    Qinvdraw = wish(Qinv,t+Q_prvar);    % draw inv q from the wishart distribution
    Qdraw = inv(Qinvdraw);              % find non-inverse q 
    
%% Draw 3: Sigma from p(Sigma|y,B_t) which is i-Wishart

    % We draw Sigma from an Inverse Wishart distribution. The parameters 
    % of the distirbution are derrived as equation Y in the main report. 
    % The mean is taken as the inverse of the sum of squared residuals
    % added to the prior mean. The variance is simply T. 
    
    % Find residuals using data and the current draw of coefficients
    resids = zeros(M,t);
    for i = 1:t
        resids(:,i) = y(:,i) - Z((i-1)*M+1:i*M,:)*Btdraw(:,i);
    end
    
    % Here we create a matrix of the accumulated sum of squared residuals, to
    % be used as the mean parameter in the i-wishart draw below. 
    sse_2S = zeros(M,M);
    for i = 1:t
        sse_2S = sse_2S + resids(:,i)*resids(:,i)';
    end
    
    Sigmainv = inv(sse_2S + Sigma_prmean);          % compute mean to use for the Wishart
    Sigmainvdraw = wish(Sigmainv,t+Sigma_prvar);    % draw from the Wishsart distribution
    Sigmadraw = inv(Sigmainvdraw);                  % turn into non-inverse Sigma
    Sigmachol = chol(Sigmadraw);                    % Cholesky decomposition (for IRF analysis)
    
%% IRF 
    % We only apply IRF analysis once we have exceeded the burn-in draws
    % phase.
    if irep > nburn;         
            %%
            %Create matrix that is going to contain all beta draws over
            %which we will take the mean after the GS as our moment
            %estimate: 
            Bt_postmean = Bt_postmean + Btdraw;
            % biga is the A matrix of the VAR(1) version of our VAR(2) model. 
            % biga is 6x6 matrix. The matrix
            % biga changes in every period of the analysis, because the
            % coefficients are time varying, so we apply the analysis below
            % in every time period. 
            
            biga = zeros(M*p,M*p); 
            for j = 1:p-1
                biga(j*M+1:M*(j+1),M*(j-1)+1:j*M) = eye(M); % fill the A matrix with identity matrix (3) in bottom left corner
            end

            % the following procedure is applied separately in each time
            % period. 
            
            % this loop takes coefficients of the relevant time period from
            % Bt_draw (which contains all coefficients for all t) and uses
            % them to update the biga matrix, so that it can change for
            % every t. 
            for i = 1:t 
                bbtemp = Btdraw(M+1:K,i);  % get the draw of B(t) at time i=1,...,T  (exclude intercept)
                splace = 0;
                for ii = 1:p
                    for iii = 1:M
                        biga(iii,(ii-1)*M+1:ii*M) = bbtemp(splace+1:splace+M,1)'; %load non-intercept coefficient draws
                        splace = splace + M;
                    end
                end
                
                % create the shock matrix 
                % we want to create a shock matrix in which the third
                % column is [0 0 1]', therefore implementing a unit shock
                % in the interest rate. 
                
                %shock = eye(3); %Unit initial shock
                shock = Sigmachol';   % First shock is the Cholesky of the VAR covariance
                diagonal = diag(diag(shock));
                shock = inv(diagonal)*shock;
                
                % Now get impulse responses for 1 through nhor future
                % periods. impresp is a 3x63 matrix which contains 9
                % response values in total for each period, 3 for each 
                % variable. These three responses correspond to the 3
                % possible shocks that are contained in the schock
                % matrix.  
                % bigai is updated through mulitiplication with the 
                % coefficient matrix after each time period. 
                
                % This chunk implements the IRF analysis. S
                
                % results matrix to store impulse responses in all periods
                impresp = zeros(M,M*nhor); 
                % Fill in the first period of the results matrix with the shock (as defined above) 
                impresp(1:M,1:M) = shock;
                % create a separate variable for the a matrix so that we
                % can update it for each period of the IRF analysis. 
                bigai = biga; 
                % This follows the impulse response function as in equation XX.
                % Fill in each period of the results matrix according to
                % the impulse response function formula. 
                for j = 1:nhor-1
                    impresp(:,j*M+1:(j+1)*M) = bigj*bigai*bigj'*shock;
                    bigai = bigai*biga; % update the coefficient matrix for next period
                end

                % The section below keeps only the responses that we are interested in:
                % - those from the periods 1975q1, 1981q3, and 1996q1
                % - those that correspond to the shock in the interest 
                % rate (i.e. those caused by the third column of our shock
                % matrix). 
    
                
                if yearlab(i,1) == 1975.00;   % store only IRF from 1975:Q1
                    impf_m = zeros(M,nhor);
                    jj=0;
                    for ij = 1:nhor
                        jj = jj + M;    % select only the third column for each time period of the IRF
                        impf_m(:,ij) = impresp(:,jj);
                    end
                    
                % for each iteration of the Gibbs sample, fill in the
                % results along the first dimension 
                    imp75(irep-nburn,:,:) = impf_m; 
                end
                if yearlab(i,1) == 1981.50;   % store only IRF from 1975:Q1
                    impf_m = zeros(M,nhor);
                    jj=0;
                    for ij = 1:nhor
                        jj = jj + M;    % select only the third column for each time period of the IRF
                        impf_m(:,ij) = impresp(:,jj);
                    end
                % for each iteration of the Gibbs sample, fill in the
                % results along the first dimension 
                    imp81(irep-nburn,:,:) = impf_m; 
                end
                if yearlab(i,1) == 1996.00;   % store only IRF from 1975:Q1
                    impf_m = zeros(M,nhor);
                    jj=0;
                    for ij = 1:nhor
                        jj = jj + M;    % select only the third column for each time period of the IRF
                        impf_m(:,ij) = impresp(:,jj);
                    end
                % for each iteration of the Gibbs sample, fill in the
                % results along the first dimension 
                    imp96(irep-nburn,:,:) = impf_m; 
                end
            end %END geting impulses for each time period 
        end %END the impulse response calculation section   
end %END main Gibbs loop (for irep = 1:nrep+nburn)
clc;
toc; % Stop timer and print total time
%% ================ END GIBBS SAMPLING ==================================
%Take the mean of the draw of the betas as moment estimate: 
Bt_postmean = Bt_postmean./nrep; 
% This section takes moments along the first dimension, i.e. across the
% Gibbs sample iterations. The moments are for the 16th, 50th and
% 84th percentile. 

    qus = [.16, .5, .84];
    imp75XY=squeeze(quantile(imp75,qus)); 
    imp81XY=squeeze(quantile(imp81,qus));
    imp96XY=squeeze(quantile(imp96,qus));
    
    % Plot impulse responses
    figure       
    set(0,'DefaultAxesColorOrder',[0 0 0],...
        'DefaultAxesLineStyleOrder','--|-|--')
    subplot(3,3,1)
    plot(1:nhor,squeeze(imp75XY(:,1,:)))
    title('Impulse response of inflation, 1975:Q1')
    xlim([1 nhor])
    ylim([-0.2 0.1])
    yline(0)
    set(gca,'XTick',0:3:nhor)
    subplot(3,3,2)
    plot(1:nhor,squeeze(imp75XY(:,2,:)))
    title('Impulse response of unemployment, 1975:Q1')
    xlim([1 nhor])
    ylim([-0.2 0.2])
    yline(0)
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,3)
    %ylim([0 1])
    yline(0)
    plot(1:nhor,squeeze(imp75XY(:,3,:)))
    title('Impulse response of interest rate, 1975:Q1')
    xlim([1 nhor])
    %ylim([-0.3 0.1])
    yline(0)
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,4)
    plot(1:nhor,squeeze(imp81XY(:,1,:)))
    title('Impulse response of inflation, 1981:Q3')
    xlim([1 nhor])
    ylim([-0.2 0.1])
    yline(0)
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,5)
    plot(1:nhor,squeeze(imp81XY(:,2,:)))
    title('Impulse response of unemployment, 1981:Q3')
    xlim([1 nhor])
    ylim([-0.2 0.2])
    yline(0)
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,6)
    plot(1:nhor,squeeze(imp81XY(:,3,:)))
    title('Impulse response of interest rate, 1981:Q3')
    xlim([1 nhor])
    %ylim([-0.4 0.1])
    yline(0)
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,7)
    plot(1:nhor,squeeze(imp96XY(:,1,:)))
    title('Impulse response of inflation, 1996:Q1')
    xlim([1 nhor])
    ylim([-0.2 0.1])
    yline(0)
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,8)
    plot(1:nhor,squeeze(imp96XY(:,2,:)))
    title('Impulse response of unemployment, 1996:Q1')
    xlim([1 nhor])
    ylim([-0.2 0.2])
    yline(0)
    set(gca,'XTick',0:3:nhor)
    subplot(3,3,9)
    plot(1:nhor,squeeze(imp96XY(:,3,:)))
    title('Impulse response of interest rate, 1996:Q1')
    xlim([1 nhor])
     %ylim([0 1])
     yline(0)
    set(gca,'XTick',0:3:nhor)
    

disp('             ')
disp('To plot impulse responses, use:         plot(1:nhor,squeeze(imp75XY(:,VAR,:)))           ')
disp('             ')
disp('where VAR=1 for impulses of inflation, VAR=2 for unemployment and VAR=3 for interest rate')



