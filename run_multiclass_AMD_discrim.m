clc 
clear
load('covtype_scale01.mat');
[n,d] = size(data);
if ~issparse(data)
    data = sparse(data);
end

parameter_matrix = [
          1,         0.5,     0.01417, 0.75, 9.90, 0.36, 70.62, 0.13 
        0.3,         0.5,   0.0388585, 0.63, 19.52, 0.87, 71.29, 0.05 
];

times = 20;  % run 20 times for calculating mean accuracy

for p = 1: size(parameter_matrix, 1)
    
    sr = RandStream.create('mt19937ar','Seed',1);
    RandStream.setGlobalStream(sr);
    errorNum = zeros(times,1);
    queryNum = zeros(times,1);
  
    % parameters
    delta = parameter_matrix(p,1);
    eta = parameter_matrix(p,2);
    b = parameter_matrix(p,3); % control the ratio of label query
    
    startime = cputime;
    
    for run = 1:times
        index = randperm(n);
        [w, N_t,errNum] = multiclass_AMD_discrim_sparse_eff(data',delta, eta, b, index);
        errorNum(run) = errNum;
        queryNum(run) = N_t /n;
    end
    
    duration = cputime - startime;
    
    accRate = (1 - errorNum./n)*100;
    accMean = mean(accRate);
    accStd = std(accRate);
    
    %-------------test model performance on test data-------------------------
    %[testAcc, testStd]= testModel(testData, allw);
    
    %-------------output result to file----------------------------------------
    fid = fopen('ttest/MD_AMD_covtype_scale01.txt','a');
    fprintf(fid,'name = covtype_scale01,  MD_AMD, runTimes= %d \n', times);
    fprintf(fid,'delta, eta, b, duration[s], query ratio + std, acc + std \n');
    fprintf(fid,'%11g, %11g, %11g, %.2f, %.2f, %.2f, ', delta, eta, b, duration/(times), 100*(mean(queryNum)), 100*(std(queryNum)));
    fprintf(fid, '%.2f, %.2f \n', accMean, accStd);
    fclose(fid);
    
    fid = fopen('ttest/MD_AMD_covtype_scale01.txt','a');
    fprintf(fid,'name = covtype_scale01 \n');
    for run = 1:times
        fprintf(fid, '%.4f \n', accRate(run));
    end
    fclose(fid);
    
end

