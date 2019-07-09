#include <mex.h>
#include <string.h>

void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]) 

{
    mwSize D;
    mwSize K;
    mwSize N;
    double nu0;
    double* h;
    double* g;
    double* newh;
    mwIndex d;
    mxArray *temp;
    double* nnzIndex;
    mwSize nnzSize;
    mwIndex j;
    mwIndex k;
    double pg;
    double newhdk;
    
    
    N = mxGetM(prhs[0]);
    
    K = mxGetN(prhs[0]);
    
    D = mxGetM(prhs[1]);
      
    
    g = mxGetPr(prhs[0]);
    h = mxGetPr(prhs[1]);
    newh = mxGetPr(prhs[2]);
    
    nu0 = mxGetScalar(prhs[3]);
    
        
    for(d = 0; d < D; d ++)
    {
         temp = mxGetCell(prhs[4],d);
         nnzIndex = mxGetPr(temp);
         nnzSize = mxGetNumberOfElements(temp);
         
         for(k = 0; k < K; k++)
         { 
             pg = 0;
             for (j = 0; j < nnzSize; j ++)
             {
                 pg += g[N * k + (int)nnzIndex[j]-1];
             }
             
             newhdk = newh[D * k + d] / (pg + nu0 * h[D*k + d]);
             
             for (j = 0; j < nnzSize; j ++)
             {                
                g[N *k + (int)nnzIndex[j]-1] *= newhdk;
             }
             
             h[D*k + d] *= newhdk;
             
             
         }
                  
    }

    
    
    
}