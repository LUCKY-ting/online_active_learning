#include "mex.h"
#include "math.h"
#include "matrix.h"
#include "stdlib.h"
#include "float.h"
#include "time.h"


#define Malloc(type,n) (type *)malloc((n)*sizeof(type))


void getDiscr_value(double *x, double *sigma, int d, int classNum, double *discr_val){
    int i, j;
    for (i = 0; i < classNum; i++)
    {
        discr_val[i] = 0;
        for (j = 0; j < d; j++)
        {
          discr_val[i] += sigma[i*d + j] * pow(x[j],2);
        }
    }
}

double max(double a, double b)
{
    if (a >= b) return a;
    else return b;
}

double uniform(double a, double b)
{
    return ((double) rand())/ RAND_MAX * (b -a) + a;
}

int binornd(double p)
{
    int x;
    double u;
    u = uniform(0.0, 1.0);
    x = (u <= p)? 1:0;
    return(x);
}

void mexFunction(int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[])
{
    if(nrhs != 5) {
    mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                      "5 inputs required.");
    }
    if(mxIsSparse(prhs[0])==0){
    mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                      "data matrix is not sparse!");
    }
    
    double *data,*miu, *index, *N_t, *x, *g_t, *sigma, *accRate, *f1score;
    double gamma, eta, b, c_t, rho_t, pred_v, discr_value, precision, recall;
    int i,j,k,p, N,d,y,pred_y, Z_t,low,high,nonzerosNum;
    int tp, pred_pos, pos, accNum;
    int * idx;
    mwIndex *ir, *jc;
    
    /*Read Input Data*/
    data = mxGetPr(prhs[0]);  // use the mxGetPr function to point to the input matrix data.
    gamma = mxGetScalar(prhs[1]);
    eta = mxGetScalar(prhs[2]);
    b = mxGetScalar(prhs[3]);
    index = mxGetPr(prhs[4]);
    
    // a column is an instance 
    d = (int)mxGetM(prhs[0]); //get Number of rows in array
    N = (int)mxGetN(prhs[0]); //get Number of columns in array
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
   
    /* preparing outputs */
    plhs[0] = mxCreateDoubleMatrix(d-1, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
    plhs[3] = mxCreateDoubleMatrix(1, 1, mxREAL);
    miu = mxGetPr(plhs[0]);
    N_t = mxGetPr(plhs[1]);
    accRate = mxGetPr(plhs[2]);
    f1score = mxGetPr(plhs[3]);

    g_t = Malloc(double,d-1);
    sigma = Malloc(double,d-1);
    for (k = 0; k < (d-1); k++)  sigma[k] = 1;
    
    srand(1); // 设置随机数种子
    accRate[0] = 0;
    tp = 0;
    pred_pos = 0;
    pos = 0;
    accNum = 0;
     /* start loop */
     for(i = 0; i < N; i++)
    {
        j = index[i] - 1;
        low = jc[j]; high = jc[j+1];
        nonzerosNum = high -1 - low;
        x = Malloc(double,nonzerosNum);
        idx = Malloc(int,nonzerosNum); // the indices of the non-zero values in x
        for (k = low; k < high -1;k++){
            x[k-low] = data[k];
            idx[k-low] = ir[k];
        }
        y = (int)data[high -1];
        
        // get the prediction
        pred_v = 0;
        for (p = 0; p < nonzerosNum; p++){
            pred_v += miu[idx[p]] * x[p];
        }
        if (pred_v > 0) 
           pred_y = 1;
        else
           pred_y = -1; 
        
        if (pred_y == y) accNum++;
        if (pred_y == 1)  pred_pos++;
        if (y == 1) pos++;
        if (pred_y == 1 && y == 1 ) tp++;
     
        discr_value = 0;
        for (p = 0; p < nonzerosNum; p++){
            discr_value += pow(x[p],2) * sigma[idx[p]];
        }
        
        c_t = eta/2 * (gamma * discr_value / (gamma + discr_value));
        rho_t = fabs(pred_v) - c_t;
          
        if (rho_t > 0) 
            Z_t = binornd(b / (b + rho_t));
        else
            Z_t = 1;

        N_t[0] = N_t[0] + Z_t; //the number of label query
        if(Z_t == 1){  // query the label y
            if (y * pred_v < 1){
                for(k = 0; k < nonzerosNum; k++){
                    g_t[idx[k]] = - y * x[k];
                    sigma[idx[k]] = sigma[idx[k]] - (sigma[idx[k]] * x[k] * x[k] * sigma[idx[k]])/(gamma + discr_value);
                    miu[idx[k]] = miu[idx[k]] - eta * (sigma[idx[k]] * g_t[idx[k]]);
                } 
            }
        }
        free(x);
        free(idx);
    } 
    free(g_t);
    free(sigma);
     
    accRate[0] = (double)accNum / N;
    precision = (double)tp / pred_pos;
    recall = (double)tp / pos;
    f1score[0] = 2 * precision * recall/ (precision + recall);
}

