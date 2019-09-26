function [bdraw] = carter_kohn_hom(y,Z,Ht,Qt,m,p,t,B0,V0)

% Output of the function: 
% bdraw: final estimates of the states (b_{t|t}), to be injected in the BS  

%Arguments of the function: 
% y = data
% Z = explanatory variable (2 lags)
% Ht = var-cov matrix of measurement equation error term
% Qt = var-cov matrix of the state equation error term 
% m = number of elements in the state vector 
% p = number of variables 
% t = number of time periods used (sample size)
% B0 = mean (prior) of the initial state vector (b_0) 
% V0 = variance (prior) of the initial state vector (V(b_0))


%% KALMAN FILTER (FF)

% Notation: mapping to our notation is the following 
% R = Sigma 
% H = Z_t
% cfe = u_t, i.e. prediction errors 
% f = F, i.e. variance of prediction errors 
% Vp = P_{t|t-1}, i.e. MSE 
% btt= b_t|t, 
% Vtt= P_t|t

%Rename priors: these are needed to initiate the loop with b_{0|0} and P_{0|0}
%and will be updated through the loop's iteration to 1,2,3,...t. 
bp = B0; % Mean of initial state vector (initial b_0|0) 
Vp = V0; % Variance of initial state vector (initial P_0|0)

% Create matrices to be filled with the filter loop:
%Store forecasted states into a matrix that has the time periods as rows
%and the number of coefficients (21) as columns: 
bt = zeros(t,m);    
%Store MSE matrix with rows being the var-cov of the coefficients and
%columns the time periods: 
Vt = zeros(m^2,t); 

R = Ht;  

%Start loop that iteates over time dimension: 
for i=1:t
    %Select Z_t from Z, which contains Z_t for all ts: 
    H = Z((i-1)*p+1:i*p,:); 
    %Compute the prediction error: observed data - predicted measurement
    %where predicted measurement is equal to (Z_t*b_{t|t})
    cfe = y(:,i) - H*bp;  
    %Compute prediction error variance: F=Z_t*P_{t|t}*Z'_t+ Sigma, and
    %compute its inverse: 
    f = H*Vp*H' + R;             
    inv_f = inv(f);
    
    %Update the states and MSE:
    %(NB: Kalman Gain not computed separately but in the formula directly)
    btt = bp + Vp*H'*inv_f*cfe;  % b_{t|t}=b_{t|t-1}+K_t(u_t)
                                 % where K_t=P_t|t*Z'_t*inv(F)
    Vtt = Vp - Vp*H'*inv_f*H*Vp; % P_{t|t}=P_{t|t-1} -K_t*Z_t*P_{t|t-1}
    
    %If iteration not at final sample period yet, then rewrite the forecasted 
    %state and MSE with the updated ones so tonbe able to continue the
    %recursions on top of the loop:
    if i < t                                
        bp = btt;  
        Vp = Vtt + Qt;       
    end
    
    %Store final states into a matrix that has t rows and the numb-
    % er of coefficients (21) as columns: 
    bt(i,:) = btt'; 
    %Store the final MSE into a matrix that has m^m rows and t columns 
    %(Thus take out the m*m matrix from Vtt for period t)
    Vt(:,i) = reshape(Vtt,m^2,1); 
end

%% Backward Sampling (BS) 

% draw Sdraw(T|T) ~ N(S(T|T),P(T|T))
%Create the output matrix to be filled with the loop: a 173*21 matrix (thus
%rows are time periods and columns the coefficients) 
bdraw = zeros(t,m); 

%-------------------------         Step 1       ---------------------------
%Start at time T by drawing beta_T from MNorm(b_T|T,P_T|T): thus fill in
%the last row of the output matrix 
bdraw(t,:) = mvnrnd(btt,Vtt,1); 

%-------------------------         Step 2       ---------------------------
%Step 2: Backward recursions
for i=1:t-1
    bf = bdraw(t-i+1,:)';         %Take out b_T, then b_{T-1}, ....
    btt = bt(t-i,:)';             %Take out b_{T-1}, then b_{T-2},....
    Vtt = reshape(Vt(:,t-i),m,m); %Take out a 21 x 21 matrix with the T-1, T-2 elements  
    f = Vtt + Qt;                 %(P_{t|t}+Q)    
    inv_f = inv(f);               %(P_{t|t}+Q)^(-1)
    cfe = bf - btt;               %(estimated beta (drawn from the distribution)-b_{t|t})
    bmean = btt + Vtt*inv_f*cfe;  % EB_t
    bvar = Vtt - Vtt*inv_f*Vtt;   % VB_t
    bdraw(t-i,:) = mvnrnd(bmean,bvar,1); %bmean' + randn(1,m)*chol(bvar); --> Draw from Mnorm(b_{t|t},P_{t|t})
end
bdraw = bdraw'; %bdraw is the final vector of joint b`T