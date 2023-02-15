
%clear; close all;
name1 = 'Unif[1,10]';
name2 = 'Mix2.2';
loss = 'op';
method = 'imp';

p = 200;
n = 20000;
X = zeros(p,n);
d = p/2;
nn = 50; % neighbors to perform os
r = 40;
theta = rand(1,n)*2*pi;
t= 1:d;
sigma = 15;
%Construct data matrix
X(1:d,:)= sin(t'*theta);
[p,n] = size(X);

[p,n] = size(X);
% construct noisy data
Y = X+sigma*randn(p,n)/sqrt(n);
X_os = zeros(p,n);
X_median = zeros(p,n);
D = pdist(Y');
D = squareform(D);
% perform local optimal shrinkage/ median
for i = 1:n
    dist_i = D(:,i);
    [~,ind] = sort(dist_i);
    Z_nn = Y(:,ind(1:nn));
    [ps,ns] = size(Z_nn);
    if ps>ns
        Z_nn = Z_nn';
        %[ps,ns] = size(Z_nn);
        %[u,s,v] = svd(Z_nn); s = diag(s);
        %s = optimal_shrinkage(s,ps/ns,loss);
        X_os_nn = optimal_shrinkage_color5(Z_nn,loss,method);
        %X_os_nn = u*diag(s)*v(:,1:ps)';
        %X_os_nn = X_os_nn';
        X_os_nn = X_os_nn';
        Z_nn = Z_nn';
    else
        %[u,s,v] = svd(Z_nn); s = diag(s);
        %s = optimal_shrinkage(s,ps/ns,loss);
        %X_os_nn = u*diag(s)*v(:,1:ps)';
        X_os_nn = optimal_shrinkage_color5(Z_nn,loss,method);
    end

     %X_os_nn = optimal_shrinkage_color5(Z_nn,loss,method);
    X_os(:,i) = X_os_nn(:,1);
    X_median(:,i) = median(Z_nn,2);
end

%global optimal shrinkage
[u,s,v] = svd(Y);
s = diag(s);
s = optimal_shrinkage(s,p/n,loss);
X0_os = u*diag(s)*v(:,1:p)';


