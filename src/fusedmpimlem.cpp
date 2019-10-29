#include "../include/csr4matrix.hpp"
#include "../include/vector.hpp"
#include "../include/profiling.h"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <mpi.h>


//Need this for MMAP??
//#define _LARGEFILE64_SOURCE 
#ifdef __APPLE__
#include <fenv.h>
#else
#include <xmmintrin.h>
#endif


struct ProgramOptions
{
    std::string mtxfilename;
    std::string infilename;
    std::string outfilename;
    int iterations;
};


struct MpiData
{
    int size;
    int rank;
};


struct Range
{
    int start;
    int end;
};


/** 
 * @brief  Simple Commandline Options Handler
 * @note   
 * @param  argc: 
 * @param  *argv[]: 
 * @retval 
 */
ProgramOptions handleCommandLine(int argc, char *argv[])
{
    if (argc != 5)
        throw std::runtime_error("wrong number of command line parameters");

    ProgramOptions progops;
    progops.mtxfilename = std::string(argv[1]);
    progops.infilename  = std::string(argv[2]);
    progops.outfilename = std::string(argv[3]);
    progops.iterations = std::stoi(argv[4]);
    return progops;
}


/** 
 * @brief  Simple implementation to initialize MPI Communication
 * @param  argc: number of command line arguments
 * @param  *argv[]: array of command line arguments
 * @retval MpiData
 */
MpiData initializeMpi(int argc, char *argv[])
{
    MpiData mpi;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);

    return mpi;
}


/** 
 * @brief  Simple Partitioner
 * @note   This partiitoner operates on the number of non-zero elements per row.
 * @param  mpi: the MPI data structure
 * @param  matrix: the input sparse matrix
 * @retval a range of on which the current MPI rank should work on. s
 */
std::vector<Range> partition(const MpiData& mpi, const Csr4Matrix& matrix)
{
    float avgElemsPerRank = (float)matrix.elements() / (float)mpi.size;

    std::cout << "Matrix elements: " << matrix.elements()
              << ", avg elements per rank: " << avgElemsPerRank << std::endl;

    std::vector<Range> ranges(mpi.size);
    int idx = 0;
    uint64_t sum = 0;
    ranges[0].start = 0;
    for (uint32_t row = 0; row < matrix.rows(); ++row) {
        sum += matrix.elementsInRow(row);
        if (sum > avgElemsPerRank * (idx + 1)) {
            ranges[idx].end = row + 1;
            idx += 1;
            ranges[idx].start = row + 1;
        }
    }
    ranges[mpi.size - 1].end = matrix.rows();

#if 0
    for (size_t i=0; i<ranges.size(); ++i) {
        std::cout << "(" << mpi.rank << "/" << mpi.size << "):"
                  << "range " << i << " from " << ranges[i].start << " to "
                  << ranges[i].end << std::endl;
    }
#endif

    return ranges;
}


/** 
 * @brief  Calculate column vector sum
 * @note   This is used to initialize the MLEM calculation.
 * @param  mpi: The MPI structure
 * @param  ranges: my ranges
 * @param  matrix: system matrix
 * @param  norm: vector to store the norm
 */
void calcColumnSums(const MpiData& mpi, const std::vector<Range>& ranges,
                    const Csr4Matrix& matrix, Vector<float>& norm)
{
    assert(matrix.columns() == norm.size());

    auto& myrange = ranges[mpi.rank];

    std::fill(norm.ptr(), norm.ptr() + norm.size(), 0);
    matrix.mapRows(myrange.start, myrange.end - myrange.start);

    for (int row =myrange.start; row < myrange.end; ++row) {
        std::for_each(matrix.beginRow2(row), matrix.endRow2(row),
                      [&](const RowElement<float>& e){ norm[e.column()] += e.value(); });
    }

#if 0
    float s = 0.0;
    for (uint i = 0; i < matrix.columns(); i++) s += norm[i];
    std::cout << "(" << mpi.rank << "/" << mpi.size << "): " <<
                 "range " << myrange.start << "-" << myrange.end << ": " << s <<std::endl;
#endif
    MPI_Allreduce(MPI_IN_PLACE, norm.ptr(), norm.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

#if 0
    s = 0.0;
    for (uint i = 0; i < matrix.columns(); i++) s += norm[i];
    std::cout << "Sum: " << std::setprecision(10) << s << std::endl;
    norm.writeToFile("norm-0.out");
#endif
}


/** 
 * @brief  Initialize the image vector
 * @note   The initial estimate is a uniform grey picture
 * @param  mpi: MPI Strucuture
 * @param  norm: the norm vector
 * @param  image: image vector
 * @param  lmsino: sinogram (input) vector
 */
void initImage(const MpiData& mpi, const Vector<float>& norm,
    Vector<float>& image, const Vector<int>& lmsino)
{
    // Sum up norm vector, creating sum of all matrix elements
    float sumnorm = 0;
    for (size_t i = 0; i < norm.size(); ++i) sumnorm += norm[i];
    float sumin = 0;
    for (size_t i = 0; i < lmsino.size(); ++i) sumin += lmsino[i];

    float initial = sumin / sumnorm;

    std::cout << "(" << mpi.rank << "/" << mpi.size << "): "
              << "init: sumnorm = " << std::setprecision(10) << sumnorm << ", sumin = " << sumin
              << ", initial value = " << initial << std::endl;

    for (size_t i = 0; i < image.size(); ++i) image[i] = initial;
}


void fused(const MpiData& mpi, const std::vector<Range>& ranges,
           const Csr4Matrix& matrix, Vector<float>& image,
           const Vector<int>& lmsino, const Vector<float>& norm
           )
{
    assert(matrix.columns() == image.size() && matrix.rows() == lmsino.size());

    Vector<float> correlation(matrix.rows(), 0);

    auto& myrange = ranges[mpi.rank];
    matrix.mapRows(myrange.start, myrange.end - myrange.start);

    profile_start_user();
    for (int row = myrange.start; row < myrange.end; ++row) {
        float res = 0;
        std::for_each(matrix.beginRow2(row), matrix.endRow2(row),
            [&](const RowElement<float>& e) { res += e.value() * image[e.column()]; });
        correlation[row] = res;
    }

    for (size_t i = 0; i < correlation.size(); ++i)
        correlation[i] = (correlation[i] != 0) ? (lmsino[i] / correlation[i]) : 0;

    Vector<float> update(image.size(), 0);

    for (int row = myrange.start; row < myrange.end; ++row) {
        std::for_each(matrix.beginRow2(row), matrix.endRow2(row),
            [&](const RowElement<float>& e) { update[e.column()] += e.value() * correlation[row]; });
    }

    profile_end_user();
    profile_start_backend();
    MPI_Allreduce(MPI_IN_PLACE, update.ptr(), update.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    profile_end_backend();

    profile_start_user();
    for (size_t i = 0; i < update.size(); ++i)
        image[i] *= (norm[i] != 0) ? (update[i] / norm[i]) : update[i];
    profile_end_user();
    
    profile_writeout();
}


/** 
 * @brief  Main MLEM function. 
 * @param  mpi: MPI Sturcture
 * @param  ranges: my ranges
 * @param  matrix: system matrix
 * @param  lmsino: sinogram vector (input)
 * @param  image: image vector (output)
 * @param  nIterations: number of iterations
 */
void mlem(const MpiData& mpi, const std::vector<Range>& ranges,
          const Csr4Matrix& matrix, const Vector<int>& lmsino,
          Vector<float>& image, int nIterations)
{
    uint32_t nColumns = matrix.columns();

    // Calculate column sums ("norm")
    Vector<float> norm(nColumns, 0);

    profile_init(REALTIME);
    profile_reset_all_timer();

    profile_start_user();
    calcColumnSums(mpi, ranges, matrix, norm);
    profile_end_user();
    profile_set_iteration(-1);
    profile_writeout();

    std::cout << "(" << mpi.rank << "/" << mpi.size << "): "
              << "calculated norm" << std::endl;

     // Fill image with initial estimate
    initImage(mpi, norm, image, lmsino);

    std::cout << "(" << mpi.rank << "/" << mpi.size << "): "
              << "Starting " << nIterations << " MLEM iterations" << std::endl;

    // Main MLEM loop
    for (int iter = 0; iter<nIterations; ++iter) {

        profile_set_iteration(iter);
        // Kernel
        fused(mpi, ranges, matrix, image, lmsino, norm);

        // Debug: sum over image values
        float s = 0;
        for (uint i = 0; i < matrix.columns(); ++i) s += image[i];

        std::cout << "(" << mpi.rank << "/" << mpi.size << "): iter: " << iter + 1
                  << ", image sum: " << std::setprecision(10) << s << std::endl;
    }
}


int main(int argc, char *argv[])
{
    ProgramOptions progops = handleCommandLine(argc, argv);

    std::cout << "Matrix file: " << progops.mtxfilename << std::endl;
    std::cout << "Input file: "  << progops.infilename << std::endl;
    std::cout << "Output file: " << progops.outfilename << std::endl;
    std::cout << "Iterations: " << progops.iterations << std::endl;

    Csr4Matrix matrix(progops.mtxfilename);

    std::cout << "Matrix rows (LORs): " << matrix.rows() << std::endl;
    std::cout << "Matrix cols (VOXs): " << matrix.columns() << std::endl;

    Vector<int> lmsino(progops.infilename);
    Vector<float> image(matrix.columns(), 0.0);

    auto mpi = initializeMpi(argc, argv);
    auto ranges = partition(mpi, matrix);

    // Flush subnormals to zero
    #ifdef __APPLE__
    fesetenv(FE_DFL_DISABLE_SSE_DENORMS_ENV);
    #else
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    #endif

    mlem(mpi, ranges, matrix, lmsino, image, progops.iterations);

    if (mpi.rank == 0) image.writeToFile(progops.outfilename);

    MPI_Finalize();
}
