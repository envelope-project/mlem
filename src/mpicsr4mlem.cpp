/**
    Copyright © 2017 Tilman Kuestner
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

#include "csr4matrix.hpp"
#include "vector.hpp"

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

#define IMG_CHECKPOINT_NAME "img.chkpt"
#define ITER_CHECKPOINT_NAME "iter.chkpt"


double wtime()
{
    struct timeval tv;
    gettimeofday(&tv, 0);

    return tv.tv_sec+1e-6*tv.tv_usec;
}


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

inline 
bool exists(const std::string& name){
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}


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
    
    fsync(image.GetFd(*myfile.rdbuf()));
    myfile.close();
}

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


MpiData initializeMpi(int argc, char *argv[])
{
    MpiData mpi;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);

    return mpi;
}


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
    for(uint i = 0; i < matrix.columns(); i++) s += norm[i];
    std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]:" <<
                 "Range " << myrange.start << "-" << myrange.end << ": " << s <<std::endl;
#endif
    MPI_Allreduce(MPI_IN_PLACE, norm.ptr(), norm.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

#ifndef MESSUNG
    s = 0.0;
    for(uint i = 0; i < matrix.columns(); i++) s += norm[i];
    std::cout << "Sum: " << std::setprecision(10) << s << std::endl;
    //norm.writeToFile("norm-0.out");
#endif
}


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

    MPI_Allreduce(MPI_IN_PLACE, result.ptr(), result.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    
    tv2 = wtime();
    *total_time += tv2-tv1;
}


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
    *comp_time += tv2-tv1;
    
    *total_time += tv2-tv1;
}


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
    
    MPI_Allreduce(MPI_IN_PLACE, update.ptr(), update.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    
    
    tv2 = wtime();
    *total_time += tv2-tv1;
}


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

    *comp_time += tv2-tv1;
    
    *total_time += tv2-tv1;
}


void mlem(const MpiData& mpi, const std::vector<Range>& ranges,
          const Csr4Matrix& matrix, const Vector<int>& lmsino,
          Vector<float>& image, int nIterations, int checkpointing)
{
    uint32_t nRows = matrix.rows();
    uint32_t nColumns = matrix.columns();
    int iter_alt = 0;

    double compute_time, total_time;

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
    // Checkpointing
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

    for (int iter = iter_alt; iter<nIterations; ++iter) {
        compute_time = 0.;
        total_time = 0.;

        calcFwProj(mpi, ranges, matrix, image, fwproj, &compute_time, &total_time);
        calcCorrel(fwproj, lmsino, correlation, &compute_time, &total_time);
        calcBkProj(mpi, ranges, matrix, correlation, update, &compute_time, &total_time);
        calcUpdate(update, norm, image, &compute_time, &total_time);

        // Debug: sum over image values
        float s = 0.0;
        for(uint i = 0; i < matrix.columns(); i++) s += image[i];
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
