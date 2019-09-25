function [bdraw,log_lik] = carter_kohn_hom(y,Z,Ht,Qt,m,p,t,B0,V0)

%Output of the function: 
% bdraw: final estimates of the states, to be used for next step of GS 
% log_lik: never used, eventually erase 

%Arguments: 
% y= dependent data
% Z= explanatory variables (2 lags)
% Ht= variance of ME error term
% Qt= variance of the SE error term 
% m= number of elements in the state vector 
% p= number of explanatory variables (inflation, un. rate, int. rate)
% t= number of time periods available (T-number of lags-training periods) 
% B0= mean of the initial state vector (a_0) 
% V0= variance of the initial state vector 

% Carter and Kohn (1994), On Gibbs sampling for state space models.

% KALMAN FILTER
%NB: mapping to our notation is the following 
% R = Sigma 
% H = Z_t
% cfe = u_t
% f = F 
% Vp = P_{t|t-1}
% btt= b_t|t
% Vtt= P_t|t

bp = B0; %mean of initial state vector (initial a_0|0) 
Vp = V0; %variance of initial state vector (initial P_0|0)
% Create matrices to be filled: 
bt = zeros(t,m);    %time period x number of coefficients matrix: for each 
                    %time period we need 21 coefficients 
Vt = zeros(m^2,t);  %MSE matrix  
log_lik = 0;
R = Ht;             % R is variance of the ME error term (sigma)
%Start loop: 
for i=1:t
    H = Z((i-1)*p+1:i*p,:);      %Take matrix Z 3 rows by 3 rows, i.e. take the x 
                                 % that you need for one period --> = Z_t
    cfe = y(:,i) - H*bp;         % conditional forecast error u_t= y_t-y_t|t-1
    f = H*Vp*H' + R;             % variance of the conditional forecast error:
                                 % F=Z_t*P_t|t*Z'_t+Sigma 
    inv_f = inv(f);              % Take the inverse of the variance 
    %log_lik = log_lik + log(det(f)) + cfe'*inv_f*cfe;
    btt = bp + Vp*H'*inv_f*cfe;  %Updating the states: b_t|t=b_t|t-1+K_t(y_t-y_t|t-1)
                                 % where K_t=P_t|t*Z'_t*inv(F)
    Vtt = Vp - Vp*H'*inv_f*H*Vp; % Updating of the MSE 
    if i < t    % If you are before the final period                            
        bp = btt;  %subsitute the state estimate with the one just computed
        Vp = Vtt + Qt; % as well as the variance             
    end
    %Store final states and MSE: 
    bt(i,:) = btt';   %store the computed final updating states 
    Vt(:,i) = reshape(Vtt,m^2,1); %store the computed final MSE: take the mxm matrix Vtt for period t 
end

% draw Sdraw(T|T) ~ N(S(T|T),P(T|T))
%Create matrices to be filled: 
bdraw = zeros(t,m); %number of periods times number of coefficients (173 x 21)

%Step 1: start at time T by drawing beta_T from MNorm(b_T|T,P_T|T)
bdraw(t,:) = mvnrnd(btt,Vtt,1); 

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
bdraw = bdraw';