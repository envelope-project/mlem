#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <nccl.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../include/cudacsr4mlem.cuh"
#include "../include/csr4matrix.hpp"
#include "../include/vector.hpp"
#include "../include/sptrans.hpp"

void csr_format_for_cuda(const Csr4Matrix& matrix, float* csrVal, unsigned long* csrRowInd, unsigned int* csrColInd);
void calcColumnSums(const Csr4Matrix& matrix, Vector<float>& norm);
void partitionMatrix(unsigned long *csr_Rows, unsigned long nnzs, unsigned int rows, unsigned int device_numbers, unsigned int *segments, unsigned int *segment_rows, unsigned int *segment_nnzs, unsigned long *offsets);
void mlem(unsigned long *csr_Rows, unsigned int *csr_Cols, float *csr_Vals, unsigned long *csr_Rows_Trans, unsigned int *csr_Cols_Trans, float *csr_Vals_Trans, int *g, float *norm, float *f, float *result_f, unsigned int rows, unsigned int cols, unsigned long nnzs, unsigned int iterations, unsigned int device_numbers, unsigned int matrix_vector_mul, unsigned int secsize_fp, unsigned int secsize_bp, unsigned int using_trans);
void help();

/*
    brief: program main body
    @param argc:    number of parameters, should be 9
    @param argv[0]: function name, determined automatically
    @param argv[1]: path of matrix file
    @param argv[2]: path of image  file
    @param argv[3]: iteration times
    @param argv[4]: number of GPUs to be used
    @param argv[5]: section size for forward  projection in NVIDIA merge-based SpMV
    @param argv[6]: section size for backward projection in NVIDIA merge-based SpMV
    @param argv[7]: whether to use transposed matrix    0: yes                 1: no
    @param argv[8]: which SpMV algorithm to use         0: merge-based SpMV    1: csr-vector SpMV

    use examples:
    ./test /scratch/pet/madpet2.p016.csr4.small /scratch/pet/Trues_Derenzo_GATE_rot_sm_200k.LMsino.small 500 2 4 4 0 0
    ./test /scratch/pet/madpet2.p016.csr4.small /scratch/pet/Trues_Derenzo_GATE_rot_sm_200k.LMsino.small 500 2 3 9 1 0
    ./test /scratch/pet/madpet2.p016.csr4.small /scratch/pet/Trues_Derenzo_GATE_rot_sm_200k.LMsino.small 500 2 3 9 0 1
    ./test /scratch/pet/madpet2.p016.csr4.small /scratch/pet/Trues_Derenzo_GATE_rot_sm_200k.LMsino.small 500 2 5 5 1 1
    ./test /scratch/pet/madpet2.p016.csr4 /scratch/pet/Trues_Derenzo_GATE_rot_sm_200k.LMsino 500 1 5 5 1 1
*/
int main(int argc, char **argv){
    if(argc != 9){
        help();
        return EXIT_FAILURE;
    }

    std::string matrixPath(argv[1]);
    std::string imagePath(argv[2]);
    unsigned int iterations          = strtol(argv[3], NULL, 10); if(iterations == 0) { help(); return EXIT_FAILURE; }  
    unsigned int device_numbers      = strtol(argv[4], NULL, 10); if(device_numbers == 0) { help(); return EXIT_FAILURE; }
    unsigned int secsize_fp          = strtol(argv[5], NULL, 10); if(secsize_fp == 0) { help(); return EXIT_FAILURE; }
    unsigned int secsize_bp          = strtol(argv[6], NULL, 10); if(secsize_bp == 0) { help(); return EXIT_FAILURE; }
    unsigned int using_trans         = strtol(argv[7], NULL, 10); if(using_trans != 0 && using_trans != 1) { help(); return EXIT_FAILURE; }
    unsigned int matrix_vector_mul   = strtol(argv[8], NULL, 10); if(matrix_vector_mul != 0 && matrix_vector_mul != 1) { help(); return EXIT_FAILURE; }
    
    int device_numbers_available = 0;
    cudaGetDeviceCount(&device_numbers_available);
    if(device_numbers_available < device_numbers){
        help();
        return EXIT_FAILURE;
    }

    // host variables
    unsigned long *csr_Rows, *csr_Rows_Trans, nnzs;
    unsigned int  *csr_Cols, *csr_Cols_Trans, rows, cols;
    int *g, sum_g = 0;
    float *csr_Vals, *csr_Vals_Trans, *f, *result_f, *norm, sum_norm = 0.0f;

    // read matrix
    Csr4Matrix matrix(matrixPath);
    rows = matrix.rows();
    cols = matrix.columns();
    nnzs = matrix.elements();
    matrix.mapRows(0, rows);    
    csr_Rows = (unsigned long*)malloc(sizeof(unsigned long) * (rows + 1));
    csr_Cols = (unsigned int*)malloc(sizeof(unsigned int) * nnzs);
    csr_Vals = (float*)malloc(sizeof(float) * nnzs);
    csr_format_for_cuda(matrix, csr_Vals, csr_Rows, csr_Cols);
    Vector<float> norm_helper(cols, 0.0);
    calcColumnSums(matrix, norm_helper);
    norm = norm_helper.ptr();
    for(unsigned int i = 0; i < cols; i++)
        sum_norm += norm[i];
    
    // read image
    Vector<int> image(imagePath);
    g = image.ptr();
    for(unsigned int i = 0; i < rows; i++)
        sum_g += g[i];
    
    // calculate initial value
    float init = sum_g / sum_norm;
    f = (float*)malloc(sizeof(float)*cols);
    result_f = (float*)malloc(sizeof(float)*cols);
    for(unsigned int i = 0; i < cols; i++)
        f[i] = init;

    // transpose matrix using algorithm ScanTrans, working on CPU
    if(using_trans == 0){
        csr_Rows_Trans = (unsigned long*) calloc (cols+1,sizeof(unsigned long));
        csr_Cols_Trans = (unsigned int*) calloc (nnzs,sizeof(unsigned int));
        csr_Vals_Trans = (float*) calloc (nnzs,sizeof(float));
        sptrans_scanTrans_specialized(rows, cols, nnzs, csr_Rows, csr_Cols, csr_Vals, csr_Cols_Trans, csr_Rows_Trans, csr_Vals_Trans);
    }

    // run MLEM
    mlem(   csr_Rows, 
            csr_Cols,
            csr_Vals,
            csr_Rows_Trans,
            csr_Cols_Trans,
            csr_Vals_Trans,
            g,
            norm,
            f, 
            result_f,
            rows,
            cols,
            nnzs,
            iterations,
            device_numbers,
            matrix_vector_mul,
            secsize_fp,
            secsize_bp,
            using_trans );
    
    // clear storage
    if (csr_Rows) free(csr_Rows);
    if (csr_Cols) free(csr_Cols);
    if (csr_Vals) free(csr_Vals);
    // if (g) free(g);
    // if (norm) free(norm);
    if (f) free(f);
    if(result_f) free(result_f);
    if(using_trans == 0){
        if (csr_Rows_Trans) free(csr_Rows_Trans);
        if (csr_Cols_Trans) free(csr_Cols_Trans);
        if (csr_Vals_Trans) free(csr_Vals_Trans);
    }

    return EXIT_SUCCESS;
}

/*
    brief: main body of MLEM algorithm
    @param csr_Rows:            row    vector in CSR format
    @param csr_Cols:            column vector in CSR format
    @param csr_Vals:            value  vector in CSR format
    @param csr_Rows_Trans:      row    vector in CSR format, for transposed matrix
    @param csr_Cols_Trans:      column vector in CSR format, for transposed matrix
    @param csr_Vals_Trans:      value  vector in CRS format, for transposed matrix 
    @param g:                   image  vector
    @param norm:                norm   vector
    @param f:                   initial values for the first MLEM iteration
    @param result_f:            output vector
    @param rows:                number of rows
    @param cols:                number of columns
    @param nnzs:                number of nnzs
    @param iterations:          iteration times
    @param device_numbers:      number of GPUs applied
    @param matrix_vector_mul:   which SpMV to use     0: merge-based SpMV   1: csr-vector SpMV
    @param secsize_fp:          section size for forward  projection (in merge-based SpMV)
    @param secsize_bp:          section size for backward projection (in merge-based SpMV)
    @param using_trans:         whether to use transposed matrix for backward projection     0: yes   1: no
*/
void mlem(  unsigned long *csr_Rows, 
            unsigned int *csr_Cols, 
            float *csr_Vals, 
            unsigned long *csr_Rows_Trans, 
            unsigned int *csr_Cols_Trans, 
            float *csr_Vals_Trans, 
            int *g, 
            float *norm, 
            float *f, 
            float *result_f, 
            unsigned int rows, 
            unsigned int cols, 
            unsigned long nnzs, 
            unsigned int iterations, 
            unsigned int device_numbers, 
            unsigned int matrix_vector_mul, 
            unsigned int secsize_fp,
            unsigned int secsize_bp,
            unsigned int using_trans )
{    
    // partition matrix
    unsigned int *segments = (unsigned int*)malloc((device_numbers+1)*sizeof(unsigned int));
    unsigned int *segment_rows = (unsigned int*)malloc(device_numbers*sizeof(unsigned int));
    unsigned int *segment_nnzs = (unsigned int*)malloc(device_numbers*sizeof(unsigned int));
    unsigned long *offsets = (unsigned long*)malloc(device_numbers*sizeof(unsigned long));
    partitionMatrix(csr_Rows, nnzs, rows, device_numbers, segments, segment_rows, segment_nnzs, offsets);

    // partition transposed matrix
    unsigned int *segments_trans;
    unsigned int *segment_rows_trans;
    unsigned int *segment_nnzs_trans;
    unsigned long *offsets_trans;
    if(using_trans == 0){
        segments_trans = (unsigned int*)malloc((device_numbers+1)*sizeof(unsigned int));
        segment_rows_trans = (unsigned int*)malloc(device_numbers*sizeof(unsigned int));
        segment_nnzs_trans = (unsigned int*)malloc(device_numbers*sizeof(unsigned int));
        offsets_trans = (unsigned long*)malloc(device_numbers*sizeof(unsigned long));
        partitionMatrix(csr_Rows_Trans, nnzs, cols, device_numbers, segments_trans, segment_rows_trans, segment_nnzs_trans, offsets_trans);
    }

    // NCCL components
    ncclComm_t *comms = (ncclComm_t*)malloc(device_numbers * sizeof(ncclComm_t));;
    cudaStream_t *streams = (cudaStream_t*)malloc(device_numbers * sizeof(cudaStream_t));
    int *devices = (int*)malloc(device_numbers * sizeof(int));    

    // device variables
    unsigned int **cuda_Rows = (unsigned int**)malloc(device_numbers*sizeof(unsigned int*));
    unsigned int **cuda_Cols = (unsigned int**)malloc(device_numbers*sizeof(unsigned int*));
    int **cuda_g = (int**)malloc(device_numbers*sizeof(int*));
    float **cuda_Vals = (float**)malloc(device_numbers*sizeof(float*));
    float **cuda_norm = (float**)malloc(device_numbers*sizeof(float*));
    float **cuda_bwproj = (float**)malloc(device_numbers*sizeof(float*));
    float **cuda_temp = (float**)malloc(device_numbers*sizeof(float*));
    float **cuda_f = (float**)malloc(device_numbers*sizeof(float*));
    unsigned int **cuda_Rows_Trans;
    unsigned int **cuda_Cols_Trans;
    float **cuda_Vals_Trans;
    if(using_trans==0){
        cuda_Rows_Trans = (unsigned int**)malloc(device_numbers*sizeof(unsigned int*));
        cuda_Cols_Trans = (unsigned int**)malloc(device_numbers*sizeof(unsigned int*));
        cuda_Vals_Trans = (float**)malloc(device_numbers*sizeof(float*));
    }

    // initialization
    unsigned int blocksize = 1024;   // unique blocksize for all kernel calls
    unsigned int *gridsize_fwproj = (unsigned int*)malloc(device_numbers*sizeof(unsigned int));
    unsigned int *gridsize_correl = (unsigned int*)malloc(device_numbers*sizeof(unsigned int));
    unsigned int *gridsize_bwproj = (unsigned int*)malloc(device_numbers*sizeof(unsigned int));
    unsigned int *gridsize_update = (unsigned int*)malloc(device_numbers*sizeof(unsigned int));
    for(unsigned int i = 0; i < device_numbers; i++){
        cudaSetDevice(i);
        cudaStreamCreate(streams+i);
        devices[i] = i;

        cudaMalloc((void**)&cuda_Rows[i], sizeof(unsigned int)*(segment_rows[i] + 1));
        cudaMalloc((void**)&cuda_Cols[i], sizeof(unsigned int)*segment_nnzs[i]);
        cudaMalloc((void**)&cuda_Vals[i], sizeof(float)*segment_nnzs[i]);
        if(using_trans == 0){
            cudaMalloc((void**)&cuda_Rows_Trans[i], sizeof(unsigned int)*(segment_rows_trans[i] + 1));
            cudaMalloc((void**)&cuda_Cols_Trans[i], sizeof(unsigned int)*segment_nnzs_trans[i]);
            cudaMalloc((void**)&cuda_Vals_Trans[i], sizeof(float)*segment_nnzs_trans[i]);
        }
        cudaMalloc((void**)&cuda_f[i], sizeof(float)*cols);
        cudaMalloc((void**)&cuda_bwproj[i], sizeof(float)*cols);
        cudaMalloc((void**)&cuda_g[i], sizeof(int)*segment_rows[i]);
        if(using_trans == 0){
            cudaMalloc((void**)&cuda_temp[i], sizeof(float)*rows);
            cudaMalloc((void**)&cuda_norm[i], sizeof(float)*segment_rows_trans[i]);
        }
        else{
            cudaMalloc((void**)&cuda_temp[i], sizeof(float)*segment_rows[i]);
            cudaMalloc((void**)&cuda_norm[i], sizeof(float)*cols);
        }
        
        
        // copy matrix from host to devices
        for(unsigned int j = segments[i]; j <= segments[i+1]; j++)
            csr_Rows[j] -= offsets[i];
        unsigned int *csr_Rows_help = (unsigned int*)malloc((segment_rows[i] + 1)*sizeof(unsigned int));
        for(unsigned int j = 0; j < segment_rows[i] + 1; j++)
            csr_Rows_help[j] = (unsigned int)csr_Rows[segments[i]+j];
        cudaMemcpy(cuda_Rows[i], csr_Rows_help, sizeof(unsigned int)*(segment_rows[i] + 1), cudaMemcpyHostToDevice);
        csr_Rows[segments[i+1]] += offsets[i];
        free(csr_Rows_help);
        cudaMemcpy(cuda_Cols[i], csr_Cols+offsets[i], sizeof(unsigned int)*segment_nnzs[i], cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_Vals[i], csr_Vals+offsets[i], sizeof(float)*segment_nnzs[i], cudaMemcpyHostToDevice);

        // copy transposed matrix from host to devices
        if(using_trans == 0){
            for(unsigned int j = segments_trans[i]; j <= segments_trans[i+1]; j++)
                csr_Rows_Trans[j] -= offsets_trans[i];
            unsigned int *csr_Rows_Trans_help = (unsigned int*)malloc((segment_rows_trans[i] + 1)*sizeof(unsigned int));
            for(unsigned int j = 0; j < segment_rows_trans[i] + 1; j++)
                csr_Rows_Trans_help[j] = (unsigned int)csr_Rows_Trans[segments_trans[i]+j];
            cudaMemcpy(cuda_Rows_Trans[i], csr_Rows_Trans_help, sizeof(unsigned int)*(segment_rows_trans[i] + 1), cudaMemcpyHostToDevice);
            csr_Rows_Trans[segments_trans[i+1]] += offsets_trans[i];
            free(csr_Rows_Trans_help);
            cudaMemcpy(cuda_Cols_Trans[i], csr_Cols_Trans+offsets_trans[i], sizeof(unsigned int)*segment_nnzs_trans[i], cudaMemcpyHostToDevice);
            cudaMemcpy(cuda_Vals_Trans[i], csr_Vals_Trans+offsets_trans[i], sizeof(float)*segment_nnzs_trans[i], cudaMemcpyHostToDevice);
        }

        // copy other vectors from host to devices
        cudaMemcpy(cuda_g[i], g+segments[i], sizeof(int)*segment_rows[i], cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_f[i], f, sizeof(float)*cols, cudaMemcpyHostToDevice);
        cudaMemset(cuda_bwproj[i], 0, sizeof(float)*cols);
        if(using_trans == 0){
            cudaMemset(cuda_temp[i], 0, sizeof(float)*rows);
            cudaMemcpy(cuda_norm[i], norm+segments_trans[i], sizeof(float)*segment_rows_trans[i], cudaMemcpyHostToDevice);
        }
        else{
            cudaMemset(cuda_temp[i], 0, sizeof(float)*segment_rows[i]);
            cudaMemcpy(cuda_norm[i], norm, sizeof(float)*cols, cudaMemcpyHostToDevice);
        }
        

        // determine grid size for all MLEM steps
        gridsize_correl[i] = ceil((double)segment_rows[i] / blocksize);
        
        if(matrix_vector_mul == 0)
            gridsize_fwproj[i] = ceil(double(segment_rows[i] + segment_nnzs[i]) / (blocksize * secsize_fp));
        else
            gridsize_fwproj[i] = ceil(double(segment_rows[i]) / 32);
        
        if(using_trans == 0){
            gridsize_update[i] = ceil((double)segment_rows_trans[i] / blocksize);
            
            if(matrix_vector_mul == 0)
                gridsize_bwproj[i] = ceil(double(segment_rows_trans[i] + segment_nnzs_trans[i]) / (blocksize * secsize_bp));
            else
                gridsize_bwproj[i] = ceil((double)segment_rows_trans[i] / 32);        
        }   
        else{
            gridsize_update[i] = ceil((double)cols / blocksize);
            gridsize_bwproj[i] = ceil((double)segment_rows[i] / 32);
        }
        
    }

    // NCCL initialization
    ncclCommInitAll(comms, device_numbers, devices);

    // MLEM iterations
    for(unsigned int iter = 0; iter < iterations; iter++){
        
        // forward projection
        for(unsigned int i = 0; i < device_numbers; i++){
            cudaSetDevice(i);
            if(matrix_vector_mul == 0)
                calcFwProj_merge_based <<< gridsize_fwproj[i], blocksize >>> (  
                    cuda_Rows[i], 
                    cuda_Cols[i], 
                    cuda_Vals[i], 
                    cuda_f[i],
                    using_trans == 0? cuda_temp[i] + segments[i] : cuda_temp[i], 
                    secsize_fp, 
                    segment_rows[i], 
                    segment_nnzs[i]);
            else
                calcFwProj_csr_vector <<< gridsize_fwproj[i], blocksize >>> (
                    cuda_Rows[i], 
                    cuda_Cols[i], 
                    cuda_Vals[i], 
                    cuda_f[i],
                    using_trans == 0 ? cuda_temp[i] + segments[i] : cuda_temp[i], 
                    segment_rows[i]);
        }

        // correlation
        for(unsigned int i = 0; i < device_numbers; i++){
            cudaSetDevice(i);
            calcCorrel <<< gridsize_correl[i], blocksize >>> (
                cuda_g[i], 
                using_trans == 0 ? cuda_temp[i] + segments[i] : cuda_temp[i], 
                segment_rows[i]);
        }

        if(using_trans == 0){
            // sum up cuda_temp over devices
            ncclGroupStart();
            for (unsigned int i = 0; i < device_numbers; i++)
                ncclAllReduce((const void*)cuda_temp[i], (void*)cuda_temp[i], rows, ncclFloat, ncclSum, comms[i], streams[i]);
            ncclGroupEnd();
            for (unsigned int i = 0; i < device_numbers; i++) {
                cudaSetDevice(i);
                cudaStreamSynchronize(streams[i]);
            }
        }

        // backward projection
        for(unsigned int i = 0; i < device_numbers; i++){
            cudaSetDevice(i);
            if(using_trans == 0){
                if(matrix_vector_mul == 0)
                    calcBwProj_merge_based <<< gridsize_bwproj[i], blocksize >>> (  
                        cuda_Rows_Trans[i], 
                        cuda_Cols_Trans[i], 
                        cuda_Vals_Trans[i], 
                        cuda_temp[i], 
                        cuda_bwproj[i] + segments_trans[i], 
                        secsize_bp, 
                        segment_rows_trans[i], 
                        segment_nnzs_trans[i]);
                else
                    calcBwProj_csr_vector <<< gridsize_bwproj[i], blocksize >>> (
                        cuda_Rows_Trans[i], 
                        cuda_Cols_Trans[i], 
                        cuda_Vals_Trans[i], 
                        cuda_temp[i], 
                        cuda_bwproj[i] + segments_trans[i], 
                        segment_rows_trans[i]);
            }
            else 
                calcBwProj_none_trans <<< gridsize_bwproj[i], blocksize >>> (
                    cuda_Rows[i], 
                    cuda_Cols[i], 
                    cuda_Vals[i],
                    cuda_temp[i], 
                    cuda_bwproj[i], 
                    segment_rows[i]);
        }

        // update
        for(unsigned int i = 0; i < device_numbers; i++){
            cudaSetDevice(i);
            if(using_trans == 0)
                calcUpdate <<< gridsize_update[i], blocksize >>> (
                    cuda_f[i] + segments_trans[i], 
                    cuda_norm[i], 
                    cuda_bwproj[i] + segments_trans[i], 
                    segment_rows_trans[i]);
            else
                calcUpdate <<< gridsize_update[i], blocksize >>> (
                    cuda_f[i], 
                    cuda_norm[i], 
                    cuda_bwproj[i], 
                    cols);
        }

        // sum up cuda_bwproj over devices and save in cuda_f
        ncclGroupStart();
        for (unsigned int i = 0; i < device_numbers; i++)
            ncclAllReduce((const void*)cuda_bwproj[i], (void*)cuda_f[i], cols, ncclFloat, ncclSum, comms[i], streams[i]);
        ncclGroupEnd();
        for (unsigned int i = 0; i < device_numbers; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
        }

        // clear cuda_bwproj
        for(unsigned int i = 0; i < device_numbers; i++){
            cudaSetDevice(i);
            cudaMemset(cuda_bwproj[i], 0, sizeof(float)*cols);
        }

        // clear cuda_temp
        for(unsigned int i = 0; i < device_numbers; i++){
            cudaSetDevice(i);
            if(using_trans == 0)
                cudaMemset(cuda_temp[i], 0, sizeof(float)*rows);
            else
                cudaMemset(cuda_temp[i], 0, sizeof(float)*segment_rows[i]);
        }
    }

    // synchronize GPUs
    for (unsigned int i = 0; i < device_numbers; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    // Result is copied to f from device 0, actually now all devices hold the same result
    cudaSetDevice(0);
    cudaMemcpy(result_f, cuda_f[0], sizeof(float)*cols, cudaMemcpyDeviceToHost);
    float sum = 0;
    for(unsigned int i = 0; i < cols; i++)
        sum += result_f[i];
    printf("\nSum f: %f\n", sum);
    

    // free all memory
    for(unsigned int i = 0; i < device_numbers; i++){
        cudaSetDevice(i);
        ncclCommDestroy(comms[i]);
        if(cuda_Rows[i]) cudaFree(cuda_Rows[i]);
        if(cuda_Cols[i]) cudaFree(cuda_Cols[i]);
        if(cuda_Vals[i]) cudaFree(cuda_Vals[i]);
        if(using_trans == 0){
            if(cuda_Rows_Trans[i]) cudaFree(cuda_Rows_Trans[i]);
            if(cuda_Cols_Trans[i]) cudaFree(cuda_Cols_Trans[i]);
            if(cuda_Vals_Trans[i]) cudaFree(cuda_Vals_Trans[i]);
        }
        if(cuda_g[i]) cudaFree(cuda_g[i]);
        if(cuda_norm[i]) cudaFree(cuda_norm[i]);
        if(cuda_bwproj[i]) cudaFree(cuda_bwproj[i]);
        if(cuda_temp[i]) cudaFree(cuda_temp[i]);
        if(cuda_f[i]) cudaFree(cuda_f[i]);
    }
    if(segments) free(segments);
    if(segment_rows) free(segment_rows);
    if(segment_nnzs) free(segment_nnzs);
    if(offsets) free(offsets);
    if(using_trans == 0){
        if(segments_trans) free(segments_trans);
        if(segment_rows_trans) free(segment_rows_trans);
        if(segment_nnzs_trans) free(segment_nnzs_trans);
        if(offsets_trans) free(offsets_trans);
    }
    if(comms) free(comms);
    if(streams) free(streams);
    if(devices) free(devices);
    if(cuda_Rows) free(cuda_Rows);
    if(cuda_Cols) free(cuda_Cols);
    if(cuda_Vals) free(cuda_Vals);
    if(using_trans == 0){
        if(cuda_Rows_Trans) free(cuda_Rows_Trans);
        if(cuda_Cols_Trans) free(cuda_Cols_Trans);
        if(cuda_Vals_Trans) free(cuda_Vals_Trans);
    }
    if(cuda_g) free(cuda_g);
    if(cuda_norm) free(cuda_norm);
    if(cuda_bwproj) free(cuda_bwproj);
    if(cuda_temp) free(cuda_temp);
    if(cuda_f) free(cuda_f);
    if(gridsize_fwproj) free(gridsize_fwproj);
    if(gridsize_correl) free(gridsize_correl);
    if(gridsize_bwproj) free(gridsize_bwproj);
    if(gridsize_update) free(gridsize_update);
}

// load matrix into CSR format
// PS: this function is provided by the Department of Informatics
void csr_format_for_cuda(const Csr4Matrix& matrix, float* csrVal, unsigned long* csrRowInd, unsigned int* csrColInd){   
    unsigned int index = 0;
    csrRowInd[index] = 0;
    unsigned int* tempIdx;
    tempIdx = (unsigned int*) malloc(sizeof(unsigned int) * matrix.rows());
    tempIdx[0] = 0;
    // !!! using openMP here will 100% lead to error in matrix
    // #pragma omp parallel for schedule (static)
    for (unsigned int row = 0; row < matrix.rows(); ++row) {
        csrRowInd[row + 1] = csrRowInd[row] + matrix.elementsInRow(row);
        index += matrix.elementsInRow(row);
        tempIdx[row + 1] = index;
    }

    #pragma omp parallel for 
    for (unsigned int row = 0; row < matrix.rows(); ++row) {
            /*
             auto it = matrix.beginRow2(row);
             unsigned int count = 0;
             unsigned int localindex = index;
             #pragma omp parallel for reduction(+:count)
             for(unsigned int i=0; i< (matrix.endRow2(row) - it); i++){
                csrVal[localindex + i] = (it+i)->value();
                csrColInd[localindex + i] = (unsigned int)((it+i)->column());
                count++;
            }
            index += count;*/
            unsigned int idx=0;
            std::for_each(matrix.beginRow2(row), matrix.endRow2(row),[&](const RowElement<float>& e){ 
                csrVal[tempIdx[row]+idx] = e.value();
                csrColInd[tempIdx[row]+idx] = (unsigned int)e.column() ;
                idx++;
            }
               // index = index + 1; }
            );
    }
}

// summation over each matrix column, useful for calculating norms
// PS: this function is provided by the Department of Informatics
void calcColumnSums(const Csr4Matrix& matrix, Vector<float>& norm){
    assert(matrix.columns() == norm.size());

    std::fill(norm.ptr(), norm.ptr() + norm.size(), 0.0);
    matrix.mapRows(0, matrix.rows());

    #pragma omp declare reduction(vec_float_plus : std::vector<float> : \
        std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<float>())) \
        initializer(omp_priv = omp_orig)
    
    std::vector<float> res(norm.size(),0);
    #pragma omp parallel for ordered reduction(vec_float_plus:res)
    for (uint32_t row=0; row<matrix.rows(); ++row) {
        std::for_each(matrix.beginRow2(row), matrix.endRow2(row),
                      [&](const RowElement<float>& e){ res[e.column()] += e.value(); });
    }
    #pragma omp parallel for 
    for(unsigned int i=0; i<norm.size(); i++){
        norm[i] = res[i];
    }

    // norm.writeToFile("norm-0.out");
}

/* 
    brief: partition matrix evenly into equally sized segments (smaller matrices), according to the number of GPUs applied
    @param csr_Rows:        row array in CSR format
    @param nnzs:            number of nnzs
    @param rows:            number of rows
    @param device_numbers:  number of GPUs applied
    @param segments:        for each segment i: segments[i] is first row, segments[i+1] is last row
    @param segment_rows:    numbers of rows in smaller matrices
    @param segment_nnzs:    numbers of nnzs in smaller matrices
    @offsets:               row offset of each smaller matrix in the original matrix
*/
void partitionMatrix(unsigned long *csr_Rows, unsigned long nnzs, unsigned int rows, unsigned int device_numbers, unsigned int *segments, unsigned int *segment_rows, unsigned int *segment_nnzs, unsigned long *offsets){
    segments[0] = 0;
    segments[device_numbers] = rows;
    unsigned int i = 0;
    double nnzs_per_segment = ((double)nnzs / (double)device_numbers);
    for(unsigned int segment = 0; segment < device_numbers; segment++){
        for(; i <= rows; i++){
            if(csr_Rows[i] >= nnzs_per_segment * segment){
                break;
            }
        }
        segments[segment] = i;
    }
    for(unsigned int segment = 0; segment < device_numbers; segment++){
        segment_rows[segment] = segments[segment+1] - segments[segment];
        segment_nnzs[segment] = (unsigned int)(csr_Rows[segments[segment+1]] - csr_Rows[segments[segment]]);
        offsets[segment] = csr_Rows[segments[segment]];
    }
}

// output help information on prompt so that the user of this program can better understand
void help(){
    printf("\nWrong parameters! Please read the following instructions:\n\n");
    printf("total amount of parameters: 9\n");
    printf("argv[1]: path of matrix file\n");
    printf("argv[2]: path of image  file\n");
    printf("argv[3]: iteration times\n");
    printf("argv[4]: number of GPUs to be used\n");
    printf("argv[5]: section size for forward  projection (merge-based SpMV)\n");
    printf("argv[6]: section size for backward projection (merge-based SpMV)\n");
    printf("argv[7]: BP uses transposed matrix?      0: yes                1: no\n");
    printf("argv[8]: which SpMV algorithm to use?    0: merge-based SpMV   1: csr-vector SpMV\n\n");
}