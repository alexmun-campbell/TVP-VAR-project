function [aols,vbar] = ts_prior(rawdat,tau,M,p)

%Outputs of the function: 
% aols = b_ols --> Vector of OLS coefficients of the measurement equation  
% vbar = v(b_ols) --> Matrix of OLS coefficients' variances 

%Arguments of the function: 
% rawdat = data 
% tau = number of training periods 
% M = number of explanatory variables 
% p = number of lags 

% Take out the first two lags from the dependent variable as you take those
% as given to initiate the OLS regression: 
yt = rawdat(p+1:tau+p,:)';

%Create K = number of coefficients to be estimated 
K = M + p*(M^2);

%% Create Z_t matrix: 
% The same loop used in the main code: 
Zt=[];
% Loop over all time periods (excluding the first two lags)
for i = p+1:tau+p 
    %Generate 3x3 identity matrix 
    ztemp = eye(M);
    %For each lag: 
    for j = 1:p; 
        xlag = rawdat(i-j,1:M); 
        xtemp = zeros(M,M*M);    
        for jj = 1:M;            
            xtemp(jj,(jj-1)*M+1:jj*M) = xlag; 
        end
        ztemp = [ztemp   xtemp]; 
    end
    Zt = [Zt ; ztemp]; 
end

%% Beta OLS computation: 

%Create empty matrices to be filled with OLS coefficients and their var-cov: 
vbar = zeros(K,K); %  K x K variance-covariance matrix of the coefficients
xhy = zeros(K,1);  %  create a column vector as long as the number of coefficients

%Start the OLS loop that iterates through the periods: 
for i = 1:tau 
    %Select explanatory variables (correct lags) for ech t: 
    zhat1 = Zt((i-1)*M+1:i*M,:); 
    %Compute Z'Z: 
    vbar = vbar + zhat1'*zhat1;  
    %Compute Z'Y:
    xhy = xhy + zhat1'*yt(:,i);  
end

%Compute (Z'Z)^(-1):
vbar = inv(vbar); 
%Compute the beta_ols: (Z'Z)^(-1)Z'Y
aols = vbar*xhy;  

%% Var(Beta_OLS) computation: 

%Create a 3x3 zero matrix
sse2 = zeros(M,M); 

%Loop over all time periods: 
for i = 1:tau 
    %Select explanatory variables (correct lags) for each t: 
    zhat1 = Zt((i-1)*M+1:i*M,:);  
    %Compute variance of error terms: E[e_t x e'_t]
    sse2 = sse2 + (yt(:,i) - zhat1*aols)*(yt(:,i) - zhat1*aols)'; 
end

%Compute var-cov matrix of the error terms e:
hbar = sse2./tau; 

%Rewrite variance variable: 
vbar = zeros(K,K);

%Fill it with var-cov: loop over time as you need to sum over the Z_t for
%each t, thus you need to be able to select the correct Z_t from the big Z
%matrix: 
for i = 1:tau
    zhat1 = Zt((i-1)*M+1:i*M,:);
    vbar = vbar + zhat1'*inv(hbar)*zhat1; % Z'Z x Inv(Var(y))^(-1)
    
% According to the variance formula, compute the inverse:  
vbar = inv(vbar);                         

end 