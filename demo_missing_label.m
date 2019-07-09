%*************************************************************************
% Matlab code for
% He Zhao, Lan Du, Wray Buntine, Piyush Rai
% "Bayesian Multi-label Learning with Sparse Features and Labels, and Label Co-occurrences," 
% in Artificial Intelligence and Statistics (AISTATS) 2018.
%
% Written by He Zhao, http://ethanhezhao.github.io/
% Copyright @ He Zhao
%*************************************************************************

K = 100;
mu_0 = 10;
nu_0 = mu_0;

dataset_name = 'bibtex';
data = load(sprintf('./data/%s.mat',dataset_name));
missing_label = load(sprintf('./data/%s_missing_label.mat',dataset_name));

is_co_label = true;

save_dir = './demo_missing_label_save/';
if ~exist(save_dir,'dir')
    mkdir(save_dir);
end

model = BMLS(K, data.X_tr, missing_label.Y_tr, ...
data.Y_tr, 200, 2500, is_co_label, mu_0, nu_0, data.X_te, data.Y_te);
save(sprintf('%s/model_label_co.mat',save_dir),'model');

is_co_label = false;

model = BMLS(K, data.X_tr, missing_label.Y_tr, ...
data.Y_tr, 200, 2500, is_co_label, false, mu_0, nu_0, data.X_te, data.Y_te);
save(sprintf('%s/model.mat',save_dir),'model');
