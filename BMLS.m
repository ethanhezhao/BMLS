%*************************************************************************
% Matlab code for
% He Zhao, Lan Du, Wray Buntine, Piyush Rai
% "Bayesian Multi-label Learning with Sparse Features and Labels, and Label Co-occurrences," 
% in Artificial Intelligence and Statistics (AISTATS) 2018.
%
% Written by He Zhao, http://ethanhezhao.github.io/
% Copyright @ He Zhao
%*************************************************************************

function model = BMLS(K, x_train, y_train, y_train_all, Burnin, Collections, ...
is_co_label, mu_0, nu_0, x_test, y_test)


if ~exist('K','var')
    K = 100;
end
if ~exist('Burnin','var')
    Burnin = 1000;
end
if ~exist('Collections','var')
    Collections = 500;
end

thinning = 10;
iterMax = Burnin+Collections;

train_N = size(x_train,1);
x_train = double(x_train);
x_train(x_train > 1) = 1;
y_train = double(y_train);
y_train(y_train > 1) = 1;
x_train = [x_train, ones(train_N,1)]; % add default label

if exist('x_test','var')
    x_test = double(x_test);
    x_test(x_test > 1) = 1;
    y_test = double(y_test);
    y_test(y_test > 1) = 1;
    x_test = [x_test, ones(size(x_test,1),1)]; % add default label
end

D = size(x_train,2);
L = size(y_train,2);

[y_train_i,y_train_l,~] = find(y_train);


beta_0 = 0.1;
r = ones(K,1)/K;
c0 = 1;
gamma0 = 1;
h = 1.0 * ones(D,K);
g = exp(x_train * log(h)); % N * K

train_theta = g';
train_phi = randg(ones(L,K) .* beta_0);
train_phi = bsxfun(@rdivide, train_phi, sum(train_phi, 1));
  
if is_co_label % are label cooccurrences used?
    y_train_all = double(y_train_all);
    co_y = y_train_all' * y_train_all;
    co_y = triu(co_y,1);
    [co_y_ii,co_y_jj,co_y_mm] = find(co_y);
end

active_feature = cell(D); % active instances per feature
for d = 1:D
    active_feature{d} = find(x_train(:,d));
end

sum_train_prob = 0;
avg_train_prob = 0;
sum_test_prob = 0;
avg_test_prob = 0;
avg_count = 0;

timing = [];

for iter=1:iterMax
    
    tic
    Rate = sum(train_phi(y_train_l,:).*train_theta(:,y_train_i)',2);
    
    M = truncated_Poisson_rnd(Rate);
    
    [train_x_k_i,train_n_k_j] = Multrnd_Matrix_mex_fast(sparse(y_train_l,y_train_i,M,L,train_N),train_phi,train_theta);    
    
    if is_co_label        
        co_train_n_k_j = Multrnd_mijk(sparse(co_y_ii,co_y_jj,co_y_mm,L,L),train_phi,r);
        train_phi = randg(beta_0 + train_n_k_j + co_train_n_k_j');
        train_phi = bsxfun(@rdivide, train_phi, sum(train_phi, 1));
        temp=sum(train_phi.*(bsxfun(@minus,sum(train_phi,1),train_phi)),1)'/2;
        r = randg(gamma0/K+ sum(co_train_n_k_j,2)/2)./(c0+temp);        
        ell = CRT_sum_mex(sum(co_train_n_k_j,2)/2,gamma0/K);
        gamma0 = randg(1e-0 + ell)/(1e-0-1/K* sum(log(max(c0./(c0+ temp),realmin))));        
        c0 = randg(1e-0 + gamma0)/(1e-0+sum(r));
    else
        train_phi = randg(beta_0 + train_n_k_j);
        train_phi = bsxfun(@rdivide, train_phi, sum(train_phi, 1));
    end
    
    c = (train_x_k_i * x_train)';
    new_h = randg(mu_0 + c);
    sample_h_rate_mex(g, h, new_h, nu_0, active_feature);
    train_theta = g';

    if is_co_label
        temp = train_n_k_j + co_train_n_k_j';
    else
        temp = train_n_k_j;
    end       
    log_q = -log(betarnd(beta_0 .* K, sum(temp,2)));  
    ell = CRT_sum_mex(temp(:),beta_0);
    beta_0 = randg(1e-2 + ell) ./ (1e-2 + sum(K .* log_q));    

        
    timing(end+1) = toc;

    train_prob = 1 - exp(- train_phi * train_theta);
    if exist('x_test','var')
        test_g = exp(x_test * log(h));
        test_theta = test_g';
        test_prob = 1 - exp(- train_phi * test_theta);
    end
    
    if iter > Burnin
        if mod(iter - Burnin,thinning) == 0
            avg_count = avg_count + 1;
            sum_train_prob = sum_train_prob + train_prob;
            avg_train_prob = sum_train_prob ./avg_count;
            if exist('x_test','var')
                sum_test_prob = sum_test_prob + test_prob;
                avg_test_prob = sum_test_prob ./ avg_count;
            end
        end
    else
       
        avg_train_prob = train_prob;
        if exist('x_test','var')
            avg_test_prob = test_prob;
        end
    end
    
    if mod(iter,10) == 0
        fprintf('%d:\n',iter);
        train_auc_roc = compute_auc(y_train, avg_train_prob);
        fprintf('train auc:%f\n',train_auc_roc);
        if exist('x_test','var')
            test_auc_roc = compute_auc(y_test, avg_test_prob);
            fprintf('test auc:%f\n',test_auc_roc);
        end
    end
end

fprintf('avg running time per iteration:%f\n',mean(timing));

model.avg_train_prob = avg_train_prob;
model.avg_test_prob = avg_test_prob;

model.train_auc_roc = train_auc_roc;
model.test_auc_roc = test_auc_roc;

model.h = h;
model.train_theta = train_theta;
model.train_phi = train_phi;

model.mu_0 = mu_0;
model.nu_0 = nu_0;

model.beta_0 = beta_0;

if is_co_label
    model.r = r;
    model.c0 = c0;
    model.gamma0 = gamma0;
end







