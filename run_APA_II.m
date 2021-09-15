clc 
clear
load('../../paper4_experiments/data_used/farm_ads.mat');
[n,d] = size(data);
if ~issparse(data)
    data = sparse(data);
end

parameter_matrix = [
      0.004,     0.01417, 0.02, 9.68, 0.43, 82.81, 0.53, 84.66, 0.47 
      0.007,   0.0754487, 0.02, 19.61, 0.46, 85.25, 0.40, 86.72, 0.39
];

times = 20;  % run 20 times for calculating mean accuracy

for p = 1: size(parameter_matrix, 1)
    
    sr = RandStream.create('mt19937ar','Seed',1);
    RandStream.setGlobalStream(sr);
    accRate = zeros(times,1);
    f1measure = zeros(times,1);
    queryRatio = zeros(times,1);
  
    % parameters
    C = parameter_matrix(p,1);
    b = parameter_matrix(p,2);% control the ratio of label query
    
    startime = cputime;
    for run = 1:times
        index = randperm(n);
        [w, N_t, acc, f1score] = PAA_II_sparse(data',C, b, index);
        accRate(run) = acc;
        f1measure(run) = f1score;
        queryRatio(run) = N_t /n;
    end
    duration = cputime - startime;
    
    %-------------test model performance on test data-------------------------
    %[testAcc, testStd]= testModel(testData, allw);
    
    %-------------output result to file----------------------------------------
    fid = fopen('ttest/APA_II_farm_ads.txt','a');
    if p == 1
        fprintf(fid,'name = farm_ads, _APA_II_sparse, runTimes= %d \n', times);
        fprintf(fid,'C, b, duration[s], query ratio + std, acc + std, f1score + std  \n');
    end
    fprintf(fid,'%11g, %11g, %.2f, %.2f, %.2f, ', C, b, duration/(times), 100*(mean(queryRatio)), 100*(std(queryRatio)));
    fprintf(fid, '%.2f, %.2f, %.2f, %.2f \n', 100*(mean(accRate)), 100*(std(accRate)),100*(mean(f1measure)), 100*(std(f1measure)));
    fclose(fid);
    
    fid = fopen('ttest/APA_II_farm_ads.txt','a');
    fprintf(fid,'name = farm_ads \n');
    for run = 1:times
        fprintf(fid, '%.4f \n', f1measure(run));
    end
    fclose(fid);
end
