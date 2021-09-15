clc 
clear
load('BaseHock.mat');
[n,d] = size(data);
if ~issparse(data)
    data = sparse(data);
end

parameter_matrix = [
          9,   0.0078125, 7.10345e-05, 0.01, 16.72, 0.44, 94.57, 0.39, 94.54, 0.39 
          3,   0.0078125, 4.05173e-05, 0.00, 18.96, 0.53, 95.36, 0.34, 95.34, 0.34 
        0.9,   0.0078125, 0.000254138, 0.00, 23.69, 0.42, 95.65, 0.27, 95.63, 0.28 
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
%         [w, N_t, acc, f1score] = D_ADA_sparse(data',delta, eta, b, index);
%          [w, N_t, acc, f1score] = D_ADA_sparse_at(data',delta, eta, b, index);    
        [w, N_t, acc, f1score] = D_AMD_sparse(data',delta, eta, b, index);
%         [w, N_t, acc, f1score] = D_AMD_sparse_at(data',delta, eta, b, index);

        accRate(run) = acc;
        f1measure(run) = f1score;
        queryRatio(run) = N_t/n;
    end
    
    duration = cputime - startime;
    
    %-------------test model performance on test data-------------------------
    %[testAcc, testStd]= testModel(testData, allw);
    
    %-------------output result to file----------------------------------------
    fid = fopen('ttest/D_AMD_sparse_BaseHock.txt','a');
    fprintf(fid,'name = BaseHock, D_AMD_sparse, runTimes= %d \n', times);
    fprintf(fid,'delta, eta, b, duration[s], query ratio + std, acc + std, f1score + std \n');
    fprintf(fid,'%11g, %11g, %11g, %.2f, %.2f, %.2f, ', delta, eta, b, duration/(times), 100*(mean(queryRatio)), 100*(std(queryRatio)));
    fprintf(fid, '%.2f, %.2f, %.2f, %.2f \n', mean(accRate)*100, std(accRate)*100, mean(f1measure)*100, std(f1measure)*100);
    fclose(fid);
    
    fid = fopen('ttest/D_AMD_sparse_BaseHock.txt','a');
    fprintf(fid,'name = BaseHock \n');
    for run = 1:times
        fprintf(fid, '%.4f \n', f1measure(run));
    end
    fclose(fid);
end

