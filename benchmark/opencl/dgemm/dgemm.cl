// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//###########################################################################
//Kernels
//###########################################################################

__kernel void dgemm(__global double *A,__global double *B, __global double *C,__global int *m,__global int *n,__global int *k,__global double *alpha,__global double *beta)
{                                                                     
   int ROW = get_global_id(1);                                 	   
   int COL = get_global_id(0);                                    	  
                                                                      
   if(ROW<(n[0]) && COL<(m[0])){                                            
   	double sum = 0.0;                                              
   	for(int i = 0;i<k[0];i++)                                         
   		sum+=(alpha[0]) * A[ROW * (k[0]) + i] * B[i*(n[0])+COL];            
   	C[ROW*(n[0])+COL] = sum + (beta[0]) * C[ROW*(n[0])+COL];                
   }                                                                  
                                                                      
}                                                                      
