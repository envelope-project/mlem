/**
    Copyright © 2019 Technische Universitaet Muenchen
    Authors: Mengdi Wang
           Apporva Gupta
           Dai Yang

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
    OTHER DEALINGS IN THE SOFTWARE.

    The above copyright notice and this permission notice shall be 
    included in all copies and/or substantial portions of the Software,
    including binaries.
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/cudacsr4mlem.cuh"

/*
	brief: calculate correlation, output saved in fwproj in-place
	@param g:		image vector
	@param fwproj:	result of forward projection / output array
	@param rows:	number of rows (equals to length of row array - 1)
*/
__global__ void calcCorrel(int *g, float *fwproj, unsigned int rows) {
	
	// !!! gridsize x blocksize >= rows
	
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < rows) 
		if(fwproj[index] != 0.0f)
			fwproj[index] =  g[index] / fwproj[index];
}

/*
	brief: calculate update, output saved in bwproj, for mlem nccl
	@param f:		input array
	@param norm:	norm array
	@param bwproj:	result of backward projection / output array
	@param cols:	number of columns
*/
__global__ void calcUpdate(float *f, float *norm, float *bwproj, unsigned int cols) {
	
	// !!!gridsize x blocksize >= cols

	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < cols) {
		if(norm[index] == 0.0f)
			bwproj[index] = f[index] * bwproj[index];
		else
			bwproj[index] = f[index] * bwproj[index] / norm[index];
	}
}

/*
	brief: calculate forward projection using NVIDIA merge-based SpMV, output saved in fwproj
	@param csr_Row:	row    array
	@param csr_Col:	column array
	@param csr_Val:	value  array	
	@param f:		f array from last iteration
	@param fwproj:	output array
	@param secSize:	section size
	@param rows:	number of rows (equals to length of row array - 1)
	@param nnzs:	number of nnzs (equals to length of val/col array)
*/
__global__ void calcFwProj_merge_based(	unsigned int *csr_Rows, 
										unsigned int *csr_Cols, 
										float *csr_Vals, 
										float *f, 
										float *fwproj,
										unsigned int secSize, 
										unsigned int rows, 
										unsigned int nnzs) {
	
	// !!!  gridsize x blocksize x sectionsize		 >= rows + nnzs
	// !!! (gridsize x blocksize - 1) x sectionsize  <  rows + nnzs
	
	merge_based_start(	csr_Rows, 
						csr_Cols, 
						csr_Vals, 
						f, 
						fwproj, 
						secSize, 
						rows, 
						nnzs);
}

/*
	brief: calculate backward projection using NVIDIA merge-based SpMV, output saved in bwproj
	@param csr_Row_Trans:	row    array of transposed matrix	
	@param csr_Col_Trans:	column array of transposed matrix
	@param csr_Val_Trans:	value  array of transposed matrix
	@param correl:			result of correlation calculation
	@param bwproj:			output array
	@param secSize:			section size
	@param cols:			number of rows of transposed matrix (columns of original matrix)
	@param nnzs:			number of nnzs (equals to length of val/col array)
*/
__global__ void calcBwProj_merge_based(	unsigned int *csr_Rows_Trans, 
										unsigned int *csr_Cols_Trans, 
										float *csr_Vals_Trans, 
										float *correl, 
										float *bwproj,
										unsigned int secSize, 
										unsigned int cols, 
										unsigned int nnzs){

	// !!!  gridsize x blocksize x sectionsize		>= cols + nnzs
	// !!! (gridsize x blocksize - 1) x sectionsize <  cols + nnzs
	
	merge_based_start(	csr_Rows_Trans, 
						csr_Cols_Trans, 
						csr_Vals_Trans, 
						correl, 
						bwproj, 
						secSize, 
						cols, 
						nnzs);
}



/*
	brief: calculate forward projection using csr-vector SpMV, output saved in fwproj
	@param csr_Rows:	row    array
	@param csr_Cols:	column array
	@param csr_Vals:	value  array
	@param f:			f array from last iteration
	@param fwproj:		output array
	@param rows:		number of rows
*/
__global__ void calcFwProj_csr_vector (	unsigned int *csr_Rows, 
													unsigned int *csr_Cols, 
													float *csr_Vals, 
													float *f, 
													float *fwproj, 
													unsigned int rows){
	mat_vec_mul_csr_vector (csr_Rows, 
							csr_Cols, 
							csr_Vals, 
							f, 
							fwproj, 
							rows);
}

/*
	brief: calculate backward projection using csr-vector SpMV, output saved in bwproj
	@param csr_Rows_Trans:	row    array of transposed matrix
	@param csr_Cols_Trans:	column array of transposed matrix
	@param csr_Vals_Trans:	value  array of transposed matrix
	@param correl:			result of correlation calculation
	@param bwproj:			output array
	@param cols:			number of rows in transposed matrix (number of columns in original matrix)
*/
__global__ void calcBwProj_csr_vector(	unsigned int *csr_Rows_Trans, 
										unsigned int *csr_Cols_Trans, 
										float *csr_Vals_Trans, 
										float *correl, 
										float *bwproj, 
										unsigned int cols){
	mat_vec_mul_csr_vector (csr_Rows_Trans, 
							csr_Cols_Trans, 
							csr_Vals_Trans, 
							correl, 
							bwproj, 
							cols);
}

/*
	brief: calculate backward projection using no transposed matrix, output saved in bwproj
	@param csr_Rows:	row    array
	@param csr_Cols:	column array
	@param csr_Vals:	value  array
	@param correl:		result of correlation calculation
	@param bwproj:		output array
	@param rows:		number of rows
	
*/
__global__ void calcBwProj_none_trans(	unsigned int *csr_Rows, 
										unsigned int *csr_Cols, 
										float *csr_Vals, 
										float *correl, 
										float *bwproj, 
										unsigned int rows){
	trans_mat_vec_mul_warp(	csr_Rows, 
							csr_Cols, 
							csr_Vals, 
							correl, 
							bwproj, 
							rows);
}

/*
	brief: helper function for merge-based SpMV: find start coordinate for each section and call merge_based_work
	@param csr_Row:	row    array
	@param csr_Val:	value  array
	@param csr_Col:	column array
	@param x:		vector to be multiplied with the matrix
	@param result:	output
	@param secSize:	section size
	@param rows:	number of rows (equals to length of row array - 1)
	@param nnzs:	number of nnzs (equals to length of val/col array)
*/
__device__ void merge_based_start(	unsigned int *csr_Rows, 
									unsigned int *csr_Cols, 
									float *csr_Vals, 
									float *x, 
									float *result,
									unsigned int secSize, 
									unsigned int rows, 
									unsigned int nnzs) {
	
	// !!!  gridsize x blocksize x sectionsize		 >= rows + nnzs
	// !!! (gridsize x blocksize - 1) x sectionsize  <  rows + nnzs

	int lefti = 0;
	int righti = rows;
	int nexti = righti / 2;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int start = index * secSize;
	int nextj = start - nexti;
	int i = 0, j = start;

	while (i != nexti) {
		i = nexti;
		j = nextj;

		// find the first coordinate (i, j) that r[i + 1] > j - 1
		if (((int)csr_Rows[i + 1]) > j - 1)
			righti = i;
		else
			lefti = i + 1;

		nexti = (lefti + righti) / 2;
		nextj = start - nexti;

		/*
			nexti = righti only happens when index of diagonal (start) is exactly number of rows + number of nnz,
			which should not happen in reality
			if (nexti = righti)
				break;
		*/
	}

	merge_based_work(	csr_Rows, 
						csr_Cols, 
						csr_Vals, 
						x, 
						result, 
						secSize, 
						rows, 
						nnzs, 
						i,
						j);
}


/*
	brief: helper function for megr-based SpMV: perform matrix-vector multiplication for each section
	@param csr_Rows:	row    array
	@param csr_Cols:	column array
	@param csr_Vals:	value  array
	@param x:			vector to be multiplied with the matrix
	@param result:		output
	@param secSize:		section size
	@param rows:		number of rows
	@param nnzs:		number of nnzs
	@param i:			x-coordinate of start point for the section
	@param j:			y-coordinate of start point for the section
*/
__device__ void merge_based_work(	unsigned int *csr_Rows, 
									unsigned int *csr_Cols, 
									float *csr_Vals, 
									float *x, 
									float *result,
									unsigned int secSize, 
									unsigned int rows, 
									unsigned int nnzs, 
									int i, 
									int j) {
	unsigned int end = i + j + secSize;
	if (end > nnzs + rows)
		end = nnzs + rows;
	float rowTimesVector = 0.0f;
	while (i + j < end) {
		if (csr_Rows[i + 1] > j) {
			rowTimesVector += csr_Vals[j] * x[csr_Cols[j]];
			j++;
		}
		else {
			// result[i++] += rowTimesVector;
			atomicAdd(result + i, rowTimesVector);
			i++;
			rowTimesVector = 0.0f;
		}
	}
	if (rowTimesVector != 0.0f)
		// result[i] += rowTimesVector;
		atomicAdd(result + i, rowTimesVector);
}

/*
	brief: helper function for csr-vector SpMV: perform matrix-vector multiplication
	@param csr_Rows:	row    array
	@param csr_Cols:	column array
	@param csr_Vals:	value  array
	@param x:			vector to be multiplied with the matrix
	@param result:		output
	@param rows:		number of rows
*/
__device__ void mat_vec_mul_csr_vector (	unsigned int *csr_Rows, 
											unsigned int *csr_Cols, 
											float *csr_Vals, 
											float *x, 
											float *result, 
											unsigned int rows){
	__shared__ float values[1024];

	unsigned int WARP_SIZE = 32;

	unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

	unsigned int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp

	unsigned int warp_id = thread_id / WARP_SIZE; // global warp index
	
	unsigned int num_warps = (blockDim.x / WARP_SIZE) * gridDim.x; // total number of active warps
	
	// one warp per row
	for (unsigned int row = warp_id; row < rows ; row += num_warps){
		unsigned int row_start = csr_Rows[row];
		unsigned int row_end = csr_Rows[row + 1];
		
		// compute running sum per thread
		values[threadIdx.x] = 0.0;
		
		for (unsigned int jj = row_start + thread_lane ; jj < row_end ; jj += WARP_SIZE)
			values[threadIdx.x] += csr_Vals[jj] * x[csr_Cols[jj]];

		// first thread writes the result
		if (thread_lane == 0){
			for (unsigned int i = 1 ; i < WARP_SIZE ; i++)
				values[threadIdx.x] += values[threadIdx.x + i];
			
			atomicAdd(result + row, values[threadIdx.x]);
		}

		__syncthreads();
	}
}

/*
	brief: helper function for backward projection using no transposed matrix: perform transposed matrix-vector multiplication
	@param csr_Rows:	row    array
	@param csr_Cols:	column array
	@param csr_Vals:	value  array
	@param x:			vector to be multiplied with the matrix
	@param result:		output
	@param rows:		number of rows
*/
__device__ void trans_mat_vec_mul_warp(	unsigned int *csr_Rows, 
										unsigned int *csr_Cols, 
										float *csr_Vals, 
										float *x, 
										float *result, 
										unsigned int rows){

	unsigned int WARP_SIZE = 32;

	unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

	unsigned int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp

	unsigned int warp_id = thread_id / WARP_SIZE; // global warp index
	
	unsigned int num_warps = (blockDim.x / WARP_SIZE) * gridDim.x; // total number of active warps
	
	for(unsigned int row = warp_id; row < rows ; row += num_warps){
		unsigned int row_start = csr_Rows[row];
		unsigned int row_end   = csr_Rows[row + 1];
		for (unsigned int i= row_start + thread_lane; i < row_end; i += WARP_SIZE)
			atomicAdd(&result[csr_Cols[i]], csr_Vals[i] * x[row]);
	}
}
