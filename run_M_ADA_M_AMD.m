clc 
clear
load('../../paper4_experiments/data_used/url_day0_p.mat');
[n,d] = size(data);
if ~issparse(data)
    data = sparse(data);
end

parameter_matrix = [
         0.3,      0.0625,    0.156258, 9.98, 0.17, 97.33, 0.36, 96.48, 0.47 
];

times = 20;  % run 20 times for calculating mean accuracy

for p = 1: size(parameter_matrix, 1)
    
    sr = RandStream.create('mt19937ar','Seed',1);
    RandStream.setGlobalStream(sr);
    accRate = zeros(times,1);
    f1measure = zeros(times,1);
    queryRatio = zeros(times,1);
  
    % parameters
    delta = parameter_matrix(p,1);
    eta = parameter_matrix(p,2);
    b = parameter_matrix(p,3); % control the ratio of label query
    
    startime = cputime;
    for run = 1:times
        index = randperm(n);
%         [w, N_t, acc, f1score] = M_ADA_sparse(data',delta, eta, b, index);
         [w, N_t, acc, f1score] = M_AMD_sparse(data',delta, eta, b, index);

        accRate(run) = acc;
        f1measure(run) = f1score;
        queryRatio(run) = N_t/n;
    end
    
    duration = cputime - startime;
    
    %-------------test model performance on test data-------------------------
    %[testAcc, testStd]= testModel(testData, allw);
    
    %-------------output result to file----------------------------------------
    fid = fopen('url_day0_p/f1score_M_AMD_sparse.txt','a');
    if p == 1
        fprintf(fid,'name = url_day0_p, M_AMD_sparse, runTimes= %d \n', times);
        fprintf(fid,'delta, eta, b, duration[s], query ratio + std, acc + std, f1score + std \n');
    end
    fprintf(fid,'%11g, %11g, %11g, %.2f, %.2f, %.2f, ', delta, eta, b, duration/(times), 100*(mean(queryRatio)), 100*(std(queryRatio)));
    fprintf(fid, '%.2f, %.2f, %.2f, %.2f \n', mean(accRate)*100, std(accRate)*100, mean(f1measure)*100, std(f1measure)*100);
    fclose(fid);
end

