#include "mex.h"
#include "math.h"
#include "matrix.h"
#include "stdlib.h"
#include "float.h"
#include "time.h"


#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

int getClassNum(double * data, int N, mwIndex *jc)
{
   int i,y,classNum;
   classNum = 0;
   for (i = 0; i < N; i++){
      y = (int)data[jc[i+1]-1];
      if (y > classNum)
          classNum = y;
   }
   return classNum;
}

void  predict(double * w, double *x, double * A_t,int d, double * pred_v)
{
    int j;
    pred_v[0] = 0;
    for (j = 0; j < d; j++)
    {
        pred_v[0] += w[j] * (1/(A_t[j] + x[j]*x[j]))* x[j];  
    }
}

double innerproduct(double *x, double *y, int d){
   int i;
   double product = 0;
   for (i = 0; i < d; i++)
       product += x[i] * y[i];
   return product;
}

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


int maxV(double * pred_v, int classNum)
{
    int p, winner;
    double maxVal = - DBL_MAX;
    for (p = 0; p < classNum; p++)
        if (pred_v[p] > maxVal){
            maxVal = pred_v[p]; 
            winner = p;
//             printf("maxVal= %f, p =%d \n", maxVal, p);
        }
    
    return winner;
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
    if(nrhs != 3) {
    mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                      "3 inputs required.");
    }
    if(mxIsSparse(prhs[0])==0){
    mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                      "data matrix is not sparse!");
    }
    
    double *data,*w, *index,*N_t,*accRate, *f1score;
    double b, precision, recall;
    int i,j,k,p, N,d,y,pred_y, Z_t,errNum,low,high;
    int tp, pred_pos, pos, accNum;
    mwIndex *ir, *jc;
    
    /*Read Input Data*/
    data = mxGetPr(prhs[0]);  // use the mxGetPr function to point to the input matrix data.
    b = mxGetScalar(prhs[1]);
    index = mxGetPr(prhs[2]);
    
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
    
    double * x = Malloc(double,d-1);
    double * pred_v = Malloc(double,1);
    double * A_t = Malloc(double,d-1);
    for (k = 0; k < (d-1); k++)  A_t[k] = 1;
    
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
        for (k = 0; k < d-1; k++)  x[k] = 0;
        low = jc[j]; high = jc[j+1];
        for (k = low; k < high -1;k++)
            x[ir[k]] = data[k];
        y = (int)data[high -1];
        
        predict(w, x, A_t, d-1, pred_v); 
        if (pred_v[0] > 0) 
           pred_y = 1;
        else
           pred_y = -1; 
        if (pred_y == y) accNum++;
        if (pred_y == 1)  pred_pos++;
        if (y == 1) pos++;
        if (pred_y == 1 && y == 1 ) tp++;


        Z_t = binornd(b / (b + fabs(pred_v[0])));
//         printf("Z_t= %d  \n", Z_t); 
        N_t[0] = N_t[0] + Z_t; //the number of label query
        if(Z_t == 1){  // query the label y
            if (pred_y != y){
                for(k = 0; k < d-1; k++){
                    w[k] = w[k] + y * x[k];
                    A_t[k] = A_t[k] + x[k] * x[k];
                } 
            }
        }     
    } 
    
//     printf("online mistakes = %d \n", errNum);
    free(x);
    free(pred_v);
    free(A_t);
    
    accRate[0] = (double)accNum / N;
    precision = (double)tp / pred_pos;
    recall = (double)tp / pos;
    f1score[0] = 2 * precision * recall/ (precision + recall);
}
