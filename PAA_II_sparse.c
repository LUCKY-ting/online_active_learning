#include "mex.h"
#include "math.h"
#include "matrix.h"
#include "stdlib.h"
#include "float.h"
#include "time.h"


#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void  predict(double * w, double *x, int d, int classNum, double * pred_v)
{
    int i, j;
    for (i = 0; i < classNum; i++)
    {
        pred_v[i] = 0;
        for (j = 0; j < d; j++)
        {
          pred_v[i] += w[i*d + j] * x[j];
        }
//         printf("i = %d, pred_v[i] = %f\n", i, pred_v[i]);
    }
}

double innerproduct(double *x, double *y, int d){
   int i;
   double product = 0;
   for (i = 0; i < d; i++)
       product += x[i] * y[i];
   return product;
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
    if(nrhs != 4) {
    mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                      "4 inputs required.");
    }
    if(mxIsSparse(prhs[0])==0){
    mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                      "data matrix is not sparse!");
    }
    
    double *data, *w, *x, *index,*N_t,*accRate, *f1score;
    double C, b, tau_t, pred_v,squarednorm, precision, recall;
    int i,j,k,p, N,d,y,pred_y,nonzerosNum, Z_t,errNum,low,high;
    int tp, pred_pos, pos, accNum;
    int * idx;
    mwIndex *ir, *jc;
    
    /*Read Input Data*/
    data = mxGetPr(prhs[0]);  // use the mxGetPr function to point to the input matrix data.
    C = mxGetScalar(prhs[1]);
    b = mxGetScalar(prhs[2]);
    index = mxGetPr(prhs[3]);
    
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
    w = mxGetPr(plhs[0]);
    N_t = mxGetPr(plhs[1]);
    accRate = mxGetPr(plhs[2]);
    f1score = mxGetPr(plhs[3]);
    
   
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
            pred_v += w[idx[p]] * x[p];
        }
        if (pred_v > 0)
            pred_y = 1;
        else
            pred_y = -1;
        
        if (pred_y == y) accNum++;
        if (pred_y == 1)  pred_pos++;
        if (y == 1) pos++;
        if (pred_y == 1 && y == 1 ) tp++;

        Z_t = binornd(b / (b + fabs(pred_v))); 
        
        N_t[0] = N_t[0] + Z_t; //the number of label query
        if(Z_t == 1){  // query the label y
            if (y * pred_v < 1){
                squarednorm = 0;  // compute the squared l2 norm
                for (p = 0; p < nonzerosNum; p++){
                    squarednorm += x[p] * x[p];
                }
                tau_t = (1 - y * pred_v) / (squarednorm + 1/(2*C));
                for(k = 0; k < nonzerosNum; k++){
                     w[idx[k]] = w[idx[k]] + tau_t * y * x[k];
                } 
            }
        }     
    } 
    
    free(x);
    free(idx);   
    accRate[0] = (double)accNum / N;
    precision = (double)tp / pred_pos;
    recall = (double)tp / pos;
    f1score[0] = 2 * precision * recall/ (precision + recall);
    
}
