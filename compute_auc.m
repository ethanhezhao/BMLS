function auc_roc = compute_auc(y,prob)

prob = prob';
[X,Y,T,auc_roc] = perfcurve(y(:),prob(:),1);

end