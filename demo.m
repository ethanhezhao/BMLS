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

model = BMLS(K, data.X_tr, data.Y_tr, [], 200, 2500, false, mu_0, ...
nu_0, data.X_te, data.Y_te);

save_dir = './demo_save';
if ~exist(save_dir,'dir')
    mkdir(save_dir);
end

save(sprintf('%s/model.mat',save_dir),'model');