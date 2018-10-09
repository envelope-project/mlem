/**
    Copyright © 2017 Technische Universitaet Muenchen
    Authors: Tilman Kuestner
           Dai Yang
           Josef Weidendorfer

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

#define _LARGEFILE64_SOURCE
#ifndef __APPLE__
#include <xmmintrin.h>
#else
#include <fenv.h>
#endif

#include "../include/csr4matrix.hpp"
#include "../include/vector.hpp"

#include <iostream>
#include <string>
#include <stdexcept>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sys/stat.h>
#include <stdio.h>
#include <sys/time.h>

#include <mpi.h>


/*
 * Compiler Switches
 * MESSUNG -> Supress human readable output and enable csv. generation
 */

#define IMG_CHECKPOINT_NAME "img.chkpt"
#define ITER_CHECKPOINT_NAME "iter.chkpt"
#define CHKPNT_INTEVALL 5

struct ProgramOptions
{
    std::string mtxfilename;
    std::string infilename;
    std::string outfilename;
    int iterations;
    int checkpointing;
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
 * @brief  Simple Time Measurement Function
 * @note   This function should not be called very often since gettimeofday ()
 * is quite expensive. Can be also changed to other time implementation such 
 * as walltime.
 * 
 * @retval Current Real Time
 */
double wtime()
{
    struct timeval tv;
    gettimeofday(&tv, 0);

    return tv.tv_sec+1e-6*tv.tv_usec;
}

/** 
 * @brief  Check if a file exists. 
 * @note   
 * @param  name: The file name.
 * @retval true, if file exists. Otherwise false. 
 */
inline 
bool exists(const std::string& name){
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

/** 
 * @brief  Create a checkpoint for the given MLEM code.
 * @note   
 * @param  mpi: the MPI Control structure
 * @param  image: Image vector
 * @param  iter: current iteration number
 * @param  img_chkpt_name: checkpoint file name for image vector
 * @param  iter_chkpt_name: checkpoint file name for the iteration number
 * @retval None
 */
void checkPointNow(
        const MpiData& mpi,
        Vector<float>& image,
        int iter,
        const std::string& img_chkpt_name,
        const std::string& iter_chkpt_name)
{
    if(mpi.rank>0) {
        return;
    }

    // Use the synchonous version to ensure that the file is flush
    image.writeToFileSync(img_chkpt_name);

    std::ofstream myfile(iter_chkpt_name,
                         std::ofstream::out | std::ofstream::binary);
    if (!myfile.good())
        throw std::runtime_error(std::string("Cannot open file ")
                                 + iter_chkpt_name
                                 + std::string("; reason: ")
                                 + std::string(strerror(errno)));
    myfile.write(reinterpret_cast<char*>(&iter), sizeof(int));
    myfile.flush();
    
    //The fsync() is required to ensure the file flush. 
    fsync(image.GetFd(*myfile.rdbuf()));
    myfile.close();
}

/** 
 * @brief Restore working vectors from a given checkpoint
 * @note   
 * @param  image: reference to the image vector
 * @param  img_chkpt_name: name to the image vector checkpoint file
 * @param  iter: reference to the iteration vector
 * @param  iter_chkpt_name: name to the iteration vector file name
 * @retval None
 */
void restore(
        Vector<float>& image,
        const std::string& img_chkpt_name,
        int& iter,
        const std::string& iter_chkpt_name)
{
    image.readFromFile(img_chkpt_name);
    
    std::ifstream myFile(iter_chkpt_name.c_str(),
                         std::ifstream::in | std::ifstream::binary);
    if(myFile.good()) {
        myFile.read(reinterpret_cast<char*>(&iter), sizeof(int));
        
    } else {
        throw std::runtime_error(
                    std::string("Cannot open file ") +
                    iter_chkpt_name +
                    std::string("; reason: ") +
                    std::string(strerror(errno)));
    }
    myFile.close();
}

/** 
 * @brief  Simple Commandline Options Handler
 * @note   
 * @param  argc: 
 * @param  *argv[]: 
 * @retval 
 */
ProgramOptions handleCommandLine(int argc, char *argv[])
{
    if (argc != 6)
        throw std::runtime_error("wrong number of command line parameters");

    ProgramOptions progops;
    progops.mtxfilename = std::string(argv[1]);
    progops.infilename  = std::string(argv[2]);
    progops.outfilename = std::string(argv[3]);
    progops.iterations = std::stoi(argv[4]);
    progops.checkpointing = std::stoi(argv[5]);
    return progops;
}

/** 
 * @brief  Simple implementation to initialize MPI Communication
 * @note   
 * @param  argc: 
 * @param  *argv[]: 
 * @retval 
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
 * @note   This partiitoner operates on the number of NZs per row. 
 * @param  mpi: the MPI data structure
 * @param  matrix: the input sparse matrix
 * @retval a range of on which the current MPI rank should work on. s
 */
std::vector<Range> partition(
        const MpiData& mpi,
        const Csr4Matrix& matrix)
{
    float avgElemsPerRank = (float)matrix.elements() / (float)mpi.size;

#ifndef MESSUNG
    std::cout << "Matrix elements: " << matrix.elements()
              << ", avg elements per rank: " << avgElemsPerRank << std::endl;
#endif
    std::vector<Range> ranges(mpi.size);
    int idx = 0;
    uint64_t sum = 0;
    ranges[0].start = 0;
    for (uint32_t row=0; row<matrix.rows(); ++row) {
        sum += matrix.elementsInRow(row);
        if (sum > avgElemsPerRank * (idx + 1)) {
            ranges[idx].end = row + 1;
            idx += 1;
            ranges[idx].start = row + 1;
        }
    }
    ranges[mpi.size - 1].end = matrix.rows();

#ifndef MESSUNG
    for (size_t i=0; i<ranges.size(); ++i) {
        std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]:"
                  << "Range " << i <<" from " << ranges[i].start << " to " << ranges[i].end << std::endl;
    }
#endif

    return ranges;
}

/** 
 * @brief Calculate Column vector sum. 
 * @note  This is used to initialize the MLEM calculation. 
 * @param  mpi: The MPI structure
 * @param  ranges: my ranges
 * @param  matrix: system matrix
 * @param  norm: vector to store the norm
 * @retval None
 */
void calcColumnSums(const MpiData& mpi, const std::vector<Range>& ranges,
                    const Csr4Matrix& matrix, Vector<float>& norm)
{
    assert(matrix.columns() == norm.size());

    auto& myrange = ranges[mpi.rank];

    std::fill(norm.ptr(), norm.ptr() + norm.size(), 0.0);
    matrix.mapRows(myrange.start, myrange.end - myrange.start);

    for (int row=myrange.start; row<myrange.end; ++row) {
        std::for_each(matrix.beginRow2(row), matrix.endRow2(row),
                      [&](const RowElement<float>& e){ norm[e.column()] += e.value(); });
    }

#ifndef MESSUNG
    float s = 0.0;
    for(unsigned i = 0; i < matrix.columns(); i++) s += norm[i];
    std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]:" <<
                 "Range " << myrange.start << "-" << myrange.end << ": " << s <<std::endl;
#endif
    MPI_Allreduce(MPI_IN_PLACE, norm.ptr(), norm.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

#ifndef MESSUNG
    s = 0.0;
    for(unsigned i = 0; i < matrix.columns(); i++) s += norm[i];
    std::cout << "Sum: " << std::setprecision(10) << s << std::endl;
    //norm.writeToFile("norm-0.out");
#endif
}

/** 
 * @brief  Initialize the image vector
 * @note   The initial estimate is a uniform grey picture
 * @param  mpi: MPI Strucuture
 * @param  norm: the norm vector
 * @param  image: image vector
 * @param  lmsino: sinogram (input) vector
 * @retval None
 */
void initImage(
        const MpiData& mpi,
        const Vector<float>& norm,
        Vector<float>& image,
        const Vector<int>& lmsino
        )
{
    // Sum up norm vector, creating sum of all matrix elements
    float sumnorm = 0.0;
    for (size_t i=0; i<norm.size(); ++i) sumnorm += norm[i];
    float sumin = 0.0;
    for (size_t i=0; i<lmsino.size(); ++i) sumin += lmsino[i];

    float initial = static_cast<float>(sumin / sumnorm);

#ifndef MESSUNG
    std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]: "
              << "INIT: sumnorm=" << std::setprecision(10) << sumnorm << ", sumin=" << sumin
              << ", initial value=" << initial << std::endl;
#endif
    for (size_t i=0; i<image.size(); ++i) image[i] = initial;
}

/** 
 * @brief  Calculate the forward projection.
 * @note   The inner for loop can also be operated using OpenMP Parallelization
 * In addition, there is no random access here into the system Matrix if the matix
 * is partitioned row-wise. Also, a MPI allReduce is used here. 
 * @param  mpi: MPI Structure
 * @param  ranges: my ranges
 * @param  matrix: system matrix
 * @param  input: input vector (sinogram)
 * @param  result: output vector
 * @param  comp_time: time and performace measurement facility
 * @param  total_time: 
 * @retval None
 */
void calcFwProj(
        const MpiData& mpi,
        const std::vector<Range>& ranges,
        const Csr4Matrix& matrix,
        const Vector<float>& input,
        Vector<float>& result,
        double* comp_time,
        double* total_time
        )
{
    // Direct access to arrays is supported e.g. const float* in = input.ptr();

    assert(matrix.columns() == input.size() && matrix.rows() == result.size());

    std::fill(result.ptr(), result.ptr() + result.size(), 0.0);
    
    
    auto& myrange = ranges[mpi.rank];
    matrix.mapRows(myrange.start, myrange.end - myrange.start);

    double tv1, tv2;
    tv1 = wtime();

    for (int row=myrange.start; row<myrange.end; ++row) {
        float res = 0.0;
        
        std::for_each(matrix.beginRow2(row), matrix.endRow2(row),
                      [&](const RowElement<float>& e) { res += (float)e.value() * input[e.column()]; });

        result[row] = res;
    }
    
    tv2 = wtime();
    *comp_time += tv2-tv1;

    std::cout << ", calcFwProj"
    << " Compute Time: " << tv2-tv1 << "s"
    << std::endl;

    MPI_Allreduce(MPI_IN_PLACE, result.ptr(), result.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    
    tv2 = wtime();
    *total_time += tv2-tv1;
}

/** 
 * @brief  Calculate the correlation vector
 * @note   
 * @param  fwproj: The result from forward projection
 * @param  input: sinogram 
 * @param  correlation: vector to store correlation
 * @param  *comp_time: Performance analytics facility
 * @param  *total_time: 
 * @retval None
 */
void calcCorrel(
        const Vector<float>& fwproj,
        const Vector<int>& input,
        Vector<float>& correlation,
        double *comp_time,
        double *total_time)
{
    assert(fwproj.size() == input.size() && input.size() == correlation.size());

    double tv1, tv2;
    tv1 = wtime();

    for (size_t i=0; i<fwproj.size(); ++i)
        correlation[i] = (fwproj[i] != 0.0) ? (input[i] / fwproj[i]) : 0.0;
    
    tv2 = wtime();
    std::cout << ", calcCorrel"
    << " Compute Time: " << tv2-tv1 << "s"
    << std::endl;

    *comp_time += tv2-tv1;
    *total_time += tv2-tv1;
}

/** 
 * @brief  Calculate the Backward projection
 * @note   The Backward projection cannot be simple converted to OpenMP, due 
 * to potential race condition (use atomic/critical/reduction). In addition, 
 * in a row-wise partitioning, random access into memory is implied.  A MPI 
 * allReduce is also triggered at the end of this function. 
 * @param  mpi: the MPI structure.
 * @param  ranges: my ranges
 * @param  matrix: system matrix
 * @param  correlation: correlation vector (result form last step)
 * @param  update: results form this function 
 * @param  comp_time: performance analytics
 * @param  total_time: 
 * @retval None
 */
void calcBkProj(
        const MpiData& mpi,
        const std::vector<Range>& ranges,
        const Csr4Matrix& matrix,
        const Vector<float>& correlation,
        Vector<float>& update,
        double* comp_time,
        double* total_time
        )
{
    assert(matrix.rows() == correlation.size() && matrix.columns() == update.size());
    auto& myrange = ranges[mpi.rank];
    double tv1, tv2;
    tv1 = wtime();

    // Initialize update with zeros
    std::fill(update.ptr(), update.ptr() + update.size(), 0.0);
    matrix.mapRows(myrange.start, myrange.end - myrange.start);
    
    for (int row=myrange.start; row<myrange.end; ++row) {
        std::for_each(matrix.beginRow2(row), matrix.endRow2(row),
                      [&](const RowElement<float>& e){ update[e.column()] += (float)e.value() * correlation[row]; });
    }

    tv2 = wtime();
    *comp_time += tv2-tv1;

    std::cout << ", calcBkProj"
    << " Compute Time: " << tv2-tv1 << "s"
    << std::endl;
    
    MPI_Allreduce(MPI_IN_PLACE, update.ptr(), update.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    
    
    tv2 = wtime();
    *total_time += tv2-tv1;
}

/** 
 * @brief  calculate the update to the image vector
 * @note   
 * @param  update: the update to be applied on the image vector
 * @param  norm: the norm vector
 * @param  image: image vector
 * @param  comp_time: performance analytics facilities
 * @param  total_time: 
 * @retval None
 */
void calcUpdate(
        const Vector<float>& update,
        const Vector<float>& norm,
        Vector<float>& image,
        double* comp_time,
        double* total_time)
{
    assert(image.size() == update.size() && image.size() == norm.size());

    double  tv1, tv2;
    tv1 = wtime();


    for (size_t i=0; i<update.size(); ++i)
        image[i] *= (norm[i] != 0.0) ? (update[i] / norm[i]) : update[i];
    
    tv2 = wtime();

    std::cout << ", calcUpdate"
    << " Compute Time: " << tv2-tv1 << "s"
    << std::endl;

    *comp_time += tv2-tv1;
    
    *total_time += tv2-tv1;
}

/** 
 * @brief  Main MLEM function. 
 * @note   
 * @param  mpi: MPI Sturcture
 * @param  ranges: my ranges
 * @param  matrix: system matrix
 * @param  lmsino: sinogram vector (input)
 * @param  image: image vector (output)
 * @param  nIterations: number of iterations
 * @param  checkpointing: Indicator for Checkpointing
 * @retval None
 */
void mlem(const MpiData& mpi, const std::vector<Range>& ranges,
          const Csr4Matrix& matrix, const Vector<int>& lmsino,
          Vector<float>& image, int nIterations, int checkpointing)
{
    uint32_t nRows = matrix.rows();
    uint32_t nColumns = matrix.columns();
    int iter_alt = 0;

    double compute_time, total_time;

    int chkpt_int = 0;

    // Allocate temporary vectors
    Vector<float> fwproj(nRows, 0.0);
    Vector<float> correlation(nRows, 0.0);
    Vector<float> update(nColumns, 0.0);

    // Calculate column sums ("norm")
    Vector<float> norm(nColumns, 0.0);

    std::chrono::time_point<std::chrono::system_clock> start, end, buffer_time, b2_time;
    start = std::chrono::system_clock::now();

    calcColumnSums(mpi, ranges, matrix, norm);

    end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;
#ifndef MESSUNG
    std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]:"
              << "Calculated norm, elapsed time: " << elapsed_seconds.count() << "s\n";
#endif    

    // Decide whether restart from checkpoint or fill intial estimates
    if(exists(IMG_CHECKPOINT_NAME) && exists(ITER_CHECKPOINT_NAME)){

        restore(image, IMG_CHECKPOINT_NAME, iter_alt, ITER_CHECKPOINT_NAME);

#ifndef MESSUNG
        std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]:"
                  << "I got Checkpoint at iternation " << iter_alt << "\n";
#endif
    }else{
        if(exists(IMG_CHECKPOINT_NAME) || exists(ITER_CHECKPOINT_NAME)){
#ifndef MESSUNG
            std::cout << "MLEM [" << mpi.rank << "/" <<  mpi.size << "]:"
                      << "omitted incomplete checkpoint";
#endif
        }
        // Fill image with initial estimate
        initImage(mpi, norm, image, lmsino);
    }
#ifndef MESSUNG
    std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]:"
              << "Starting " << nIterations-iter_alt << " MLEM iterations" << std::endl;
#endif

    // Beginning of the Main MLEM Loop
    for (int iter = iter_alt; iter<nIterations; ++iter) {
        compute_time = 0.;
        total_time = 0.;

        // Kernel
        calcFwProj(mpi, ranges, matrix, image, fwproj, &compute_time, &total_time);
        calcCorrel(fwproj, lmsino, correlation, &compute_time, &total_time);
        calcBkProj(mpi, ranges, matrix, correlation, update, &compute_time, &total_time);
        calcUpdate(update, norm, image, &compute_time, &total_time);

        // Debug: sum over image values
        float s = 0.0;
        for(unsigned i = 0; i < matrix.columns(); i++) s += image[i];
#ifndef MESSUNG
        std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]: Iter: "
                  << iter + 1 << ", "
                  << " Image sum: " << std::setprecision(10) << s
                  << " Elapsed time: " << total_time << "s("
                  << compute_time << "s/"
                  << total_time - compute_time << "s)"
                  << std::endl;
#else
        printf("%d, %lf, %lf, %lf\n",
               iter+1,
               total_time,
               compute_time,
               total_time - compute_time);
#endif
        // Calculate time consumed for chkpointing
        if(checkpointing==1 && mpi.rank == 0){
            double tv1, tv2;
            double tchk;

            if(chkpt_int != CHKPNT_INTEVALL){
                chkpt_int++;
                continue;
            }

            tv1 = wtime();
            checkPointNow(
                        mpi,
                        image,
                        iter+1,
                        IMG_CHECKPOINT_NAME,
                        ITER_CHECKPOINT_NAME);

            tv2 = wtime();

            tchk = tv2-tv1;
#ifndef MESSUNG
            std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]: "<<
                         "Checkpointing time: " << tchk << "s\n";
#else
            printf("-255, %lf\n", tchk);
#endif
        chkpt_int = 0;
        }
        
    }
}


int main(int argc, char *argv[])
{
    ProgramOptions progops = handleCommandLine(argc, argv);
#ifndef MESSUNG
    std::cout << "Matrix file: " << progops.mtxfilename << std::endl;
    std::cout << "Input file: "  << progops.infilename << std::endl;
    std::cout << "Output file: " << progops.outfilename << std::endl;
    std::cout << "Iterations: " << progops.iterations << std::endl;
    std::cout << "Checkpointing: " << progops.checkpointing << std::endl;
#endif
    Csr4Matrix matrix(progops.mtxfilename);
#ifndef MESSUNG
    std::cout << "Matrix rows (LORs): " << matrix.rows() << std::endl;
    std::cout << "Matrix cols (VOXs): " << matrix.columns() << std::endl;
#endif
    Vector<int> lmsino(progops.infilename);
    Vector<float> image(matrix.columns(), 0.0);

    auto mpi = initializeMpi(argc, argv);

    auto ranges = partition(mpi, matrix);

    #ifndef __APPLE__
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);    
    #else
    fesetenv(FE_DFL_DISABLE_SSE_DENORMS_ENV);
    #endif

    mlem(mpi, ranges, matrix, lmsino, image, progops.iterations, progops.checkpointing);

    if (mpi.rank == 0)
        image.writeToFile(progops.outfilename);


    // Remove chekcpoint
    if (progops.checkpointing == 1 && mpi.rank == 0){
        remove(IMG_CHECKPOINT_NAME);
        remove(ITER_CHECKPOINT_NAME);
    }

    MPI_Finalize();

    return 0;
}
