function [bdraw] = carter_kohn_hom(y,Z,Ht,Qt,K,M,t,B0,V0)

% Output of the function: 
% bdraw: final estimates of the states (b_{t|t}), to be injected in the BS  

%Arguments of the function: 
% y = data
% Z = explanatory variable (2 lags)
% Ht = var-cov matrix of measurement equation error term
% Qt = var-cov matrix of the state equation error term 
% K = number of elements in the state vector 
% M = number of variables 
% t = number of time periods used (sample size)
% B0 = mean (prior) of the initial state vector (b_0) 
% V0 = variance (prior) of the initial state vector (V(b_0))


%% KALMAN FILTER (FF)

% Notation: mapping to report's notation is the following 
% R = Sigma 
% H = Z_t
% cfe = u_t, i.e. prediction errors 
% f = F, i.e. variance of prediction errors 
% Vp = P_{t|t-1}, i.e. MSE 
% btt= b_t|t, 
% Vtt= P_t|t

%Create vectors for the states and their MSE that will be updated through 
%the filter at each time t. 
%Set initial values to $b_0|0 and P_0|0: 

bp = B0; % Mean (prior) of initial state vector (initial b_0|0)) 
Vp = V0; % Variance of (prior) initial state vector (initial P_0|0))

%Create matrices to be filled with the filter loop:
%Store forecasted states into a matrix that has 
%- t rows
%- number of coefficients (21) columns 
bt = zeros(t,K);  

% Store MSE matrix with:
% - rows being the var-cov of the coefficients and
% - columns the time periods: 
Vt = zeros(K^2,t); 

R = Ht;  

%Start loop that iterates over time dimension: 
for i=1:t
    %Select independent variables for time period t from big matrix Z, 
    % which contains all observations: 
    H = Z((i-1)*M+1:i*M,:);
    
    %COMPUTE PREDICTION ERROR: 
    % observed data - predicted measurement, where predicted measurement is 
    % equal to (Z_t*b_{t|t})
    cfe = y(:,i) - H*bp;  
    
    %COMPUTE PREDICTION ERROR VARIANCE: 
    %F=Z_t*P_{t|t}*Z'_t+ Sigma, and compute its inverse: 
    f = H*Vp*H' + R;             
    inv_f = inv(f);
    
    %UPDATE STATES AND MSE:
    %(NB: Kalman Gain not computed separately but in the formula directly)
    btt = bp + Vp*H'*inv_f*cfe;  % b_{t|t}=b_{t|t-1}+K_t(u_t)
                                 % where K_t=P_t|t*Z'_t*inv(F)
    Vtt = Vp - Vp*H'*inv_f*H*Vp; % P_{t|t}=P_{t|t-1} -K_t*Z_t*P_{t|t-1}
    
    %If iteration not at final sample period yet, then rewrite the forecasted 
    %state and MSE with the updated ones so to be able to continue the
    %recursions on top of the loop:
    if i < t                                
        bp = btt;  
        Vp = Vtt + Qt;       
    end
    
    %Store final states into a matrix that has t rows and coefficients in 
    % each of the columns : 
    bt(i,:) = btt'; 
    %Store the final MSE into a matrix that has K^2 rows and t columns 
    %(Thus take out the m*m matrix from Vtt for period t)
    Vt(:,i) = reshape(Vtt,K^2,1); 
end

%% Backward Sampling (BS) 

%Create the output matrix to be filled with the loop: a 173*21 matrix (thus
%rows being time periods and columns coefficients) 
bdraw = zeros(t,K); 

%-------------------------------- Step 1 ----------------------------------
%Start at time T by drawing beta_T from MNorm(b_T|T,P_T|T): thus fill in
%the last row of the output matrix 
bdraw(t,:) = mvnrnd(btt,Vtt,1); 

%-------------------------------- Step 2 ----------------------------------
%Backward recursions
%Loop over all time periods until the penultimate one, for which you already
%computed the final draw in Step 1: 
for i=1:t-1
    bf = bdraw(t-i+1,:)';         %Take out b_T, then b_{T-1}, ....b_{2}
    btt = bt(t-i,:)';             %Take out b_{T-1}, then b_{T-2},....b_{1}
    %Take out the P_{t|t} 21 x 21 matrix for each t=T-1,T-2,...
    Vtt = reshape(Vt(:,t-i),K,K); 
    f = Vtt + Qt;                 %(P_{t|t}+Q)    
    inv_f = inv(f);               %(P_{t|t}+Q)^(-1)
    %Compute (b_{t+1}-b_t|t): 
    cfe = bf - btt;
    %Compute the moments of the distribution from which to recursively draw
    %the next states: 
    bmean = btt + Vtt*inv_f*cfe;  % EB_t
    bvar = Vtt - Vtt*inv_f*Vtt;   % VB_t
    %Draw the states: 
    bdraw(t-i,:) = mvnrnd(bmean,bvar,1); 
end
%Store the final joint vector of states as a  matrix of drawn/filtered 
%coefficients. Transpose it so that in the end: 
% - rows = coefficients 
% - columns= time periods 
bdraw = bdraw'; %bdraw is the final vector of joint b`T



