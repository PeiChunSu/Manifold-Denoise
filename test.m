
clear; close all;
name1 = 'Unif[1,10]';
name2 = 'Mix2.2';
loss = 'op';
method = 'imp';
k = 10; % times of random projection
n = 200;
p = 20000;
nn = 50; % neighbors
r = 40;
theta = 0:(2*pi/p):2*pi;
t= 1:n;
X = [];
sigma = 10;
for i = 1:length(theta)
    X = [X;sin(theta(i)*t)];
end
X = X';
[p,n] = size(X);
Q = randn(2000,p);
X = Q*X;
[p,n] = size(X);
%{
cm = 10;
N = trnd(cm,p,n)./(sqrt(1.25)*sqrt(n));

Q1 = diag(ones(1,p));%
Q2 = diag(ones(1,n));%
%[Q1,~] = qr(randn(p,p));% modified QR decomposition
%[Q2,~] = qr(randn(n,n));
[~,fT1] = Noise(p,p,name1,Q1);
[~,fT2] = Noise(n,n,name2,Q2);
A = Q1*fT1*Q1';
B = Q2*fT2*Q2';
sigma1 = sqrt(sum(diag(fT1).^2)/p);
sigma2 = sqrt(sum(diag(fT2).^2)/n);
A = A./sigma1; B = B./sigma2;
Y  = X + sigma*A*N*B;
%}
Y = X+randn(p,n)/sqrt(p);
nn_c = [];
X_os = zeros(p,n);

for i = 1:n
    ind_c = [];
    Z = [Y(:,i),Y(:,[max(1,i-200):(i-1),(i+1):min(i+200,n)])];
    for j = 1:k
        P = randn(r,p);
        S = P*Z;
        D = pdist(S');
        D = squareform(D);
        D = D(1,:);
        D(1) = inf;
        [~,ind] = sort(D);
        ind_c = [ind_c;ind];
    end
    
    A = [1,mode(ind_c(:,1:nn))];
    nb = unique(A);
    Z_nn = Z(:,nb);
    [ps,ns] = size(Z_nn);
    if ps>ns
        Z_nn = Z_nn';
        [ps,ns] = size(Z_nn);
        [u,s,v] = svd(Z_nn); s = diag(s);
        s = optimal_shrinkage(s,ps/ns,loss);
        X_os_nn = u*diag(s)*v(:,1:ps)';
        X_os_nn = X_os_nn';
    else
        [u,s,v] = svd(Z_nn); s = diag(s);
        s = optimal_shrinkage(s,ps/ns,loss);
        X_os_nn = u*diag(s)*v(:,1:ps)';
    end

     %X_os_nn = optimal_shrinkage_color5(Z_nn,loss,method);
    X_os(:,i) = X_os_nn(:,1);
end
[u,s,v] = svd(Y);
s = diag(s);
s = optimal_shrinkage(s,p/n,loss);
X0_os = u*diag(s)*v(:,1:p)';

% compare global optimal shrinkage nbs and local optimal shrinkage nbs
% TP FP FN TN F1
% How to find true nbs
% Transition: curvature large nbs small vs p/n effect
% Compare Manifold denoise literatures

