#ifndef __STRUCTURES_H__
#define __STRUCTURES_H__

#include <string>
#include <vector>

/*
Used when the data is further split on an mpi process 
depending on the number of devices connected to it.
*/
struct further_split{
	int id;
	size_t start = 0;
	size_t end = 0;
	size_t nnz = 0;
	size_t num_rows = 0;
	size_t free_bytes = 0;
	size_t total_bytes = 0;
};

/*
Used to time each iteration in the code
*/
struct TimingIter
{
	float fwproj_time = 0.0;
	float fwproj_redc_time = 0.0;
	float corr_calc_time = 0.0;
	float update_calc_time = 0.0;
	float update_redc_time = 0.0;
	float img_update_time = 0.0;
	float img_sum_time = 0.0;
};

/*
Used to time the one time part of the code
along with the iterations 
*/
struct TimingRuntime
{
	float further_par_time = 0.0;
	float struct_to_csr_vector_time = 0.0;
	float alloc_copy_to_d_time = 0.0;
	float norm_calc_time = 0.0;
	float norm_redc_time = 0.0;
	float calc_setting_image_time = 0.0;
	std::vector<TimingIter> timing_loop;
	TimingRuntime(int iter){
		timing_loop.resize(iter);
	}
};

/*
Struct to save the arguments from the command line
*/
struct ProgramOptions
{
    std::string mtxfilename;
    std::string infilename;
    std::string outfilename;
    int iterations;
    int checkpointing;
};

/*
Used to save the size and rank of each mpi process
*/
struct MpiData
{
    int size;
    int rank;
};

/*
Used to save the start and end row for each mpi process
*/
struct Range
{
    int start;
    int end;
};

#endif