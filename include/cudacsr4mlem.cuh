
#ifndef _KERNEL_CUH_ 
#define _KERNEL_CUH_


// correlation and update
__global__ void calcCorrel(int *g, float *fwproj, unsigned int rows);
__global__ void calcUpdate(float *f, float *norm, float *bwproj, unsigned int cols);

// forward/backward projection implementing NVIDIA merge-based SpMV 
__global__ void calcFwProj_merge_based(unsigned int *csr_Rows, unsigned int *csr_Cols, float *csr_Vals, float *f, float *fwproj, unsigned int secSize, unsigned int rows, unsigned int nnzs);
__global__ void calcBwProj_merge_based(unsigned int *csr_Rows_Trans, unsigned int *csr_Cols_Trans, float *csr_Vals_Trans, float *correl, float *bwproj, unsigned int secSize, unsigned int cols, unsigned int nnzs);

// forward/backward projection implementing csr-vector SpMV
__global__ void calcFwProj_csr_vector(unsigned int *csr_Rows, unsigned int *csr_Cols, float *csr_Vals, float *f, float *fwproj, unsigned int rows);
__global__ void calcBwProj_csr_vector(unsigned int *csr_Rows_Trans, unsigned int *csr_Cols_Trans, float *csr_Vals_Trans, float *correl, float *bwproj, unsigned int cols);

// backward projection using no transposed matrix
__global__ void calcBwProj_none_trans(unsigned int *csr_Rows, unsigned int *csr_Cols, float *csr_Vals, float *correl, float *bwproj, unsigned int rows);




// helper functions for NVIDIA merge-based SpMV
__device__ void merge_based_start(unsigned int *csr_Rows, unsigned int *csr_Cols, float *csr_Vals, float *x, float *result, unsigned int secSize, unsigned int rows, unsigned int nnzs);
__device__ void merge_based_work (unsigned int *csr_Rows, unsigned int *csr_Cols, float *csr_Vals, float *x, float *result, unsigned int secSize, unsigned int rows, unsigned int nnzs, int i, int j);

// helper function for csr-vector SpMV
__device__ void mat_vec_mul_csr_vector(unsigned int *csr_Rows, unsigned int *csr_Cols, float *csr_Vals, float *x, float *result, unsigned int rows);

// helper function for backward projection using no transposed matrix
__device__ void trans_mat_vec_mul_warp(unsigned int *csr_Rows, unsigned int *csr_Cols, float *csr_Vals, float *x, float *result, unsigned int rows);

#endif