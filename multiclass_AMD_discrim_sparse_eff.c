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
    if(nrhs != 5) {
    mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                      "5 inputs required.");
    }
    if(mxIsSparse(prhs[0])==0){
    mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                      "data matrix is not sparse!");
    }
    
    double *data,*w, *s_t, *index, *H_t, *N_t,*x,*errNum;
    double delta, eta, b, firstMax_value, secondMax_value, margin, value, pred_y, cur_loss, q_t, discr_winner, discr_runner, product;
    int i,j,k,N,p,d,classNum,y,winner, runner,runnerup, Z_t,low,high, nonzerosNum;
    mwIndex *ir, *jc;
    int * idx;
    /*Read Input Data*/
    data = mxGetPr(prhs[0]);  // use the mxGetPr function to point to the input matrix data.
    delta = mxGetScalar(prhs[1]);
    eta = mxGetScalar(prhs[2]);
    b = mxGetScalar(prhs[3]);
    index = mxGetPr(prhs[4]);
    
    // a column is an instance 
    d = (int)mxGetM(prhs[0]); //get Number of rows in array
    N = (int)mxGetN(prhs[0]); //get Number of columns in array
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    classNum = getClassNum(data, N, jc);
    
//     printf("Number of instances= %d, dim = %d, classNum = %d\n", N, d, classNum);
    
    s_t = Malloc(double,(d-1)*classNum);
    H_t = Malloc(double,(d-1)*classNum);
    for (k = 0; k < (d-1)*classNum; k++){
        H_t[k] = delta;
        s_t[k]= 0;
    }
    
    /* preparing outputs */
    plhs[0] = mxCreateDoubleMatrix(d-1, classNum, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
    w = mxGetPr(plhs[0]);
    N_t = mxGetPr(plhs[1]);
    errNum =  mxGetPr(plhs[2]);
    
    double * pred_v = Malloc(double,classNum);
    double * discr_value = Malloc(double,classNum);
    srand(1); // 设置随机数种子
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
        y = (int)data[high -1] - 1;
        
        for (k = 0; k < classNum; k++){ // get the prediction from each class
            pred_v[k] = 0;
            for (p = 0; p < nonzerosNum; p++){
                pred_v[k] += w[k*(d-1) + idx[p]] * x[p];
            }
        }
        winner = maxV(pred_v,classNum);
        firstMax_value = pred_v[winner];
        
        if (winner != y) errNum[0] = errNum[0] + 1 ;

        pred_v[winner] = - DBL_MAX;
        runner = maxV(pred_v,classNum);
        secondMax_value = pred_v[runner];
        margin = firstMax_value - secondMax_value;
        
        product = 0;
        for (p = 0; p < nonzerosNum; p++) // compute the inner product
             product += x[p] * x[p];
        if(product < 1) product = 1; // to avoid dividing zero
        
        for (k = 0; k < classNum; k++){ //compute the discriminate value
            discr_value[k] = 0;
            for (p = 0; p < nonzerosNum; p++){
                discr_value[k] += 1 / H_t[k*(d-1) + idx[p]] * pow(x[p],2);
            }
            discr_value[k] = discr_value[k] / product;
        }
        discr_winner = discr_value[winner];
        discr_value[winner] = - DBL_MAX;
        discr_runner = discr_value[maxV(discr_value, classNum)];
        q_t = margin - eta/2 * ( discr_winner + discr_runner);
        if (q_t > 0)
            Z_t = binornd(b / (b + q_t));
        else
            Z_t = 1;
        
        N_t[0] = N_t[0] + Z_t; //the number of label query
        if(Z_t == 1){  // query the label y
            if (y == winner){
                runnerup = runner;
                value = secondMax_value;
                pred_y = firstMax_value;
            }else{
                runnerup = winner;
                value = firstMax_value;
                pred_y = pred_v[y]; // the prediction from the true-class weight
            }
            cur_loss = max(0, 1 + value - pred_y);
            
            if (cur_loss > 0){
                for(k = 0; k < nonzerosNum; k++){
                   s_t[runnerup * (d-1) + idx[k]] = sqrt(pow(s_t[runnerup*(d-1) + idx[k]], 2) + pow(x[k], 2));
                   H_t[runnerup * (d-1) + idx[k]] = delta + s_t[runnerup*(d-1) + idx[k]];
                   w[runnerup*(d-1)+ idx[k]] = w[runnerup*(d-1) + idx[k]] - eta * (1/ H_t[runnerup*(d-1) + idx[k]] ) * x[k];
                    
                   s_t[y * (d-1) + idx[k]] = sqrt(pow(s_t[y*(d-1) + idx[k]], 2) + pow(x[k], 2));
                   H_t[y * (d-1) + idx[k]] = delta + s_t[y*(d-1) + idx[k]];
                   w[y*(d-1)+ idx[k]] = w[y*(d-1)+ idx[k]] + eta * (1/ H_t[y*(d-1)+ idx[k]] ) * x[k];
                } 
            }
        }
        free(x);
        free(idx);
    } 
//     printf("online mistakes = %d \n", errNum);
    free(pred_v);
    free(discr_value);
    free(s_t);
    free(H_t);
      
}