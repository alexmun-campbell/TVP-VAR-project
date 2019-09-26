function [aols,vbar] = ts_prior(rawdat,tau,p,plag)

%Outputs of the function: 
% aols = b_ols --> Vector of OLS coefficients of the VAR model 
% vbar = v(b_ols) --> Matrix of OLS coefficeints variances 

%Arguments of the function: 
% rawdat = data 
% tau = tnumber of training periods 
% p = number of explanatory variables 
% plag = number of lags 

% Take out the first two lags from the dependent variable as you take those
% as given to initiate the OLS regression: 
yt = rawdat(plag+1:tau+plag,:)';

%Create m = number of coefficients to be estimated 
m = p + plag*(p^2);

%% Create Z_t matrix: 
% The same loop used in the main code is used, with the only difference
% that when computing 
Zt=[];
% Loop over all time periods (excluding the first two lags)
for i = plag+1:tau+plag 
    %Generate 3x3 identity matrix 
    ztemp = eye(p);
    %For each lag: 
    for j = 1:plag; 
        xlag = rawdat(i-j,1:p); 
        xtemp = zeros(p,p*p);    
        for jj = 1:p;            
            xtemp(jj,(jj-1)*p+1:jj*p) = xlag; 
        end
        ztemp = [ztemp   xtemp]; 
    end
    Zt = [Zt ; ztemp]; 
end

%% Beta OLS computation: 
%Create empty matrixes to be filled with OLS coefficients and their var-cov: 
vbar = zeros(m,m); % mxm variance-covariance matrix of the coefficients
xhy = zeros(m,1);  % create a column vector as big as the number of coefficients
%Start the OLS loop that iterates through the periods: 
for i = 1:tau 
    %Select explanatory variables (correct lags) for ech t: 
    zhat1 = Zt((i-1)*p+1:i*p,:); 
    %Compute Z'Z: 
    vbar = vbar + zhat1'*zhat1;  
    %Compute Z'Y
    xhy = xhy + zhat1'*yt(:,i);  
end

%Compute (Z'Z)^(-1):
vbar = inv(vbar); 
%Compute the beta_ols: (Z'Z)^(-1)Z'Y
aols = vbar*xhy;  

%% Var(Beta_OLS) computation: 

%Create a 3x3 zero matrix
sse2 = zeros(p,p); 
%Loop over all time periods: 
for i = 1:tau 
    %Select explanatory variables (correct lags) for ech t: 
    zhat1 = Zt((i-1)*p+1:i*p,:);  
    %Compute variance of error terms: E[e_te'_t]
    sse2 = sse2 + (yt(:,i) - zhat1*aols)*(yt(:,i) - zhat1*aols)'; 
end
%Compute var-cov matrix of the error terms e:
hbar = sse2./tau; 

%Rewrite variance variable: 
vbar = zeros(m,m);
%Fill it with var-cov: loop over time as you need to sum over the Z_t for
%each t, thus you need to be able to select the correct Z_t from the big Z
%matrix: 
for i = 1:tau
    zhat1 = Zt((i-1)*p+1:i*p,:);
    vbar = vbar + zhat1'*inv(hbar)*zhat1; % X'X x Inv(Var(y))-1 
% Compute the inverse to respect formula of the variance: 
vbar = inv(vbar);                         

end 