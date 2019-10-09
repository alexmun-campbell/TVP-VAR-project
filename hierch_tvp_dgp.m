function [x, beta_t,theta_t] = hierch_tvp_dgp()


if nargin==0
    T = 200;
    M = 3;
    p = 1;
    K = p*(M^2);

    Q = 0.01*eye(K);
    R = 0.001*eye(K);
    Sigma = [1.0000   -0.5000   -0.2500;
            -0.5000    1.2500   -0.3750;
            -0.2500   -0.3750    1.3125];
    
    theta_t = zeros(T+p,K);
    beta_t = zeros(T+p,K);
    A_0 = eye(K);
    
    theta_0 = [0.7  0     0.35;
               0    0.7   0 ;
               0    0.65  0.7;];
    for i=1:T+p
        if i==1
            theta_t(i,:) = theta_0(:)' +  randn(1,K)*chol(R);      
        else
            theta_t(i,:) = theta_t(i-1,:) + randn(1,K)*chol(R);      
        end        
    end
    
    for i=1:T+p
        beta_t(i,:) = theta_t(i,:)*A_0 + randn(1,K)*chol(Q);        
    end
end

y = [rand(p,M); zeros(T,M)];
for i = p+1:T+p
    y(i,:) = y(i-1,:)*reshape(beta_t(i,:),M,M) + randn(1,M)*chol(Sigma);
end

x=y(p+1:T,:);