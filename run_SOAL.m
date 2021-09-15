clc 
clear
load('../../paper4_experiments/data_used/url_day0_p.mat');
[n,d] = size(data);
if ~issparse(data)
    data = sparse(data);
end

parameter_matrix = [
         40,     0.03125,    0.179696, 0.06, 9.81, 0.17, 95.94, 0.21, 94.63, 0.28 
          9,   0.0441942,     0.66016, 0.08, 19.84, 0.34, 96.27, 0.09, 95.06, 0.13 
];

times = 20;  % run 20 times for calculating mean accuracy

for p = 1: size(parameter_matrix, 1)
    
    sr = RandStream.create('mt19937ar','Seed',1);
    RandStream.setGlobalStream(sr);
    accRate = zeros(times,1);
    f1measure = zeros(times,1);
    queryRatio = zeros(times,1);
  
    % parameters
    gamma = parameter_matrix(p,1);
    eta = parameter_matrix(p,2);
    b = parameter_matrix(p,3); % control the ratio of label query
    startime = cputime;
    
    for run = 1:times
        index = randperm(n);
        [w, N_t, acc, f1score] = SOAL_sparse(data',gamma, eta, b, index);
        accRate(run) = acc;
        f1measure(run) = f1score;
        queryRatio(run) = N_t /n;
    end
    duration = cputime - startime;
    

    %-------------test model performance on test data-------------------------
    %[testAcc, testStd]= testModel(testData, allw);
    
    %-------------output result to file----------------------------------------
    fid = fopen('ttest/SOAL_url_day0_p.txt','a');
    fprintf(fid,'name = url_day0_p, SOAL_sparse, runTimes= %d \n', times);
    fprintf(fid,'gamma, eta, b, duration[s], query ratio + std, acc + std, f1score + std  \n');
    fprintf(fid,'%11g, %11g, %11g, %.2f, %.2f, %.2f, ', gamma, eta, b, duration/(times), 100*(mean(queryRatio)), 100*(std(queryRatio)));
    fprintf(fid, '%.2f, %.2f, %.2f, %.2f \n', 100*(mean(accRate)), 100*(std(accRate)),100*(mean(f1measure)), 100*(std(f1measure)));
    fclose(fid);

    fid = fopen('ttest/SOAL_url_day0_p.txt','a');
    fprintf(fid,'name = url_day0_p \n');
    for run = 1:times
        fprintf(fid, '%.4f \n', f1measure(run));
    end
    fclose(fid);
end
