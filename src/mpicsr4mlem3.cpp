/**
    Copyright © 2019 Technische Universitaet Muenchen
    Authors: Tilman Kuestner
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

#include <memory>
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
#include <vector>

#include <mpi.h>

#include "../include/csr4matrix.hpp"
#include "../include/vector.hpp"


#ifdef _OMP_
    #include <omp.h>
#endif

#ifdef XEON_PHI
    #include <hbwmalloc.h>
    #define MALLOC(type, num) (type*) hbw_malloc((num) * sizeof(type))
    #define FREE(x) hbw_free(x)
#else
    #define MALLOC(type, num) (type*) malloc((num) * sizeof(type))
    #define FREE(x) free(x)
#endif


struct ProgramOptions
{
    std::string mtxfilename;
    std::string infilename;
    std::string outfilename;
    int iterations;
    int numberOfThreads;  // number of threads per MPI rank
    int nCBlocks; // number of blocks (per thread) due to cache blocking
};


struct Range
{
    int startRow;
    int endRow;
};


struct CSR
{
    int nRowIndex, nColumn;
    int nElements;
    uint32_t startRow, endRow;
    uint32_t *columns;
    uint64_t *rowIndex;
    float *values;
};


struct MpiData
{
    int size, rank, numberOfThreads;
    int nCBlocks;

    uint64_t numberOfElements = 0;

    Range range;

    #ifdef _OMP_
        std::vector<std::vector<CSR>> threadsCSR;
    #else
        CSR* mpiCSR;
    #endif
};


ProgramOptions handleCommandLine(int argc, char *argv[])
{
    if (argc != 7)
        throw std::runtime_error("wrong number of command line parameters");

    ProgramOptions progops;
    progops.mtxfilename = std::string(argv[1]);
    progops.infilename  = std::string(argv[2]);
    progops.outfilename = std::string(argv[3]);
    progops.iterations = std::stoi(argv[4]);
    progops.numberOfThreads = std::stoi(argv[5]);
    progops.nCBlocks = std::stoi(argv[6]);
    return progops;
}


MpiData initializeMpiOmp(int argc, char *argv[], int numberOfThreads, int nCBlocks)
{
    MpiData mpi;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);

    mpi.nCBlocks = 1;

    #ifdef _OMP_
        omp_set_num_threads(numberOfThreads);
        mpi.numberOfThreads = numberOfThreads;
        mpi.nCBlocks = nCBlocks;
    #endif

    return mpi;
}


void calcColumnSums(const MpiData& mpi, float* norm, const int normSize)
{
    #ifdef _OMP_
    for (int i = 0; i < mpi.numberOfThreads; ++i) {  // threads
        for (int j = 0; j < mpi.nCBlocks; ++j) {  // blocks
            for (int k = 0; k < mpi.threadsCSR[i][j].nColumn; ++k) {  // columns
                norm[mpi.threadsCSR[i][j].columns[k]] += mpi.threadsCSR[i][j].values[k];
            }
        }
    }
    #else
    for (size_t j = 0; j < mpi.mpiCSR->nColumn; ++j) {
        norm[mpi.mpiCSR->columns[j]] += mpi.mpiCSR->values[j];
    }
    #endif

    MPI_Allreduce(MPI_IN_PLACE, norm, normSize, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}


void initImage(const MpiData& mpi, float* norm, Vector<float>& image,
    const Vector<int>& lmsino, const size_t normSize)
{
    double sumnorm = 0.0, sumin = 0.0;
    float initial;
    size_t i = 0;

    // Sum up norm vector, creating sum of all matrix elements
    for (i = 0; i < normSize; ++i) sumnorm += norm[i];
    for (i = 0; i < lmsino.size(); ++i) sumin += lmsino[i];

    initial = static_cast<float>(sumin / sumnorm);
    for (i = 0; i < image.size(); ++i) image[i] = initial;

    #ifndef MESSUNG
        std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]: "
            << "INIT: sumnorm=" << sumnorm << ", sumin=" << sumin
            << ", initial value=" << initial << std::endl;
    #endif
}


#ifdef _OMP_
void calcFwProj(const MpiData& mpi, const Vector<float>& input,
    float* fwproj, const int fwprojSize)
{
    // Fill with zeros
    std::fill(fwproj, fwproj + fwprojSize, 0.0);

    #pragma omp parallel
    {
        const int tidx = omp_get_thread_num();
        for (int j = 0; j < mpi.nCBlocks; ++j) {  // blocks
            auto& csr = mpi.threadsCSR[tidx][j];
            for (uint32_t r=csr.startRow; r<csr.endRow; ++r) {  // r: global row index
                uint32_t rr = r - csr.startRow;  // rr: local row index
                float res = 0;
                #pragma unroll (16)
                for (uint64_t c=csr.rowIndex[rr]; c<csr.rowIndex[rr + 1]; ++c) {
                    res += csr.values[c] * input[csr.columns[c]];
                }
                fwproj[r] += res;
            }
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, fwproj, fwprojSize, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}

#else

void calcFwProj(const MpiData& mpi, const Vector<float>& input,
    float* fwproj, const int fwprojSize)
{
    float res = 0.0;

    uint32_t i, j, startColumnIndex, endColumnIndex;
    uint32_t length = mpi.mpiCSR->nRowIndex;
    uint32_t row = mpi.mpiCSR->startRow;

    // Fill with zeros
    std::fill(fwproj, fwproj + fwprojSize, 0.0);

    for(i = 1, startColumnIndex = 0;
        row < mpi.mpiCSR->endRow && i < length;
        ++i, ++row, startColumnIndex = endColumnIndex) {

        endColumnIndex = startColumnIndex +
            (mpi.mpiCSR->rowIndex[i] - mpi.mpiCSR->rowIndex[i-1]);
        res = 0.0;

        for(j = startColumnIndex; j < endColumnIndex; ++j)
            res += mpi.mpiCSR->values[j] * input[mpi.mpiCSR->columns[j]];

        fwproj[row] = res;
    }

    MPI_Allreduce(MPI_IN_PLACE, fwproj, fwprojSize, MPI_FLOAT, MPI_SUM,
        MPI_COMM_WORLD);
}
#endif


void calcCorrel(float* fwproj, const Vector<int>& input, float* correlation,
    const int fwprojSize)
{
    for (int i = 0; i < fwprojSize; ++i)
        correlation[i] = (fwproj[i] != 0.0) ? (input[i] / fwproj[i]) : 0.0;
}


#ifdef _OMP_
void calcBkProj(const MpiData& mpi, float* correlation, float* update,
    const size_t updateSize)
{
    // Fill with zeros
    std::fill(update, update + updateSize, 0.0);

    #pragma omp parallel
    {
        const int tidx = omp_get_thread_num();
        Vector<float> private_update(updateSize, 0);

        for (int j = 0; j < mpi.nCBlocks; ++j) {  // blocks
            auto& csr = mpi.threadsCSR[tidx][j];
            for (uint32_t r=csr.startRow; r<csr.endRow; ++r) {  // r: global row index
                uint32_t rr = r - csr.startRow;  // rr: local row index
                #pragma vector aligned
                #pragma ivdep
                for (uint64_t c=csr.rowIndex[rr]; c<csr.rowIndex[rr + 1]; ++c) {
                    private_update[csr.columns[c]] += csr.values[c] * correlation[r];
                }
            }
        }

        #pragma omp critical
        {
            for (size_t i = 0; i < updateSize; ++i) update[i] += private_update[i];
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, update, updateSize, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}

#else

void calcBkProj(const MpiData& mpi, float* correlation, float* update,
    const size_t updateSize)
{
    // Fill with zerons
    std::fill(update, update + updateSize, 0.0);

    uint32_t i, j, startColumnIndex, endColumnIndex;
    uint32_t length = mpi.mpiCSR->nRowIndex;
    uint32_t row = mpi.mpiCSR->startRow;

    for(i = 1, startColumnIndex = 0;
        row < mpi.mpiCSR->endRow && i < length;
        ++i, ++row, startColumnIndex = endColumnIndex) {
        endColumnIndex = startColumnIndex +
            (mpi.mpiCSR->rowIndex[i] - mpi.mpiCSR->rowIndex[i-1]);

        for(j = startColumnIndex; j < endColumnIndex; ++j)
            update[mpi.mpiCSR->columns[j]] +=
                mpi.mpiCSR->values[j] * correlation[row];
    }

    MPI_Allreduce(MPI_IN_PLACE, update, updateSize, MPI_FLOAT, MPI_SUM,
        MPI_COMM_WORLD);
}
#endif


void calcUpdate(float* update, float* norm, Vector<float>& image,
    const size_t updateSize)
{
    for (size_t i = 0; i < updateSize; ++i)
        image[i] *= (norm[i] != 0.0) ? (update[i] / norm[i]) : update[i];
}


void mlem(const MpiData& mpi, const Vector<int>& lmsino, Vector<float>& image,
    const int nIterations, const uint32_t nRows, const uint32_t nColumns)
{
    size_t i = 0;
    double sum = 0;
    double startTime, endTime, gElapsed, gBkProj, gFwProj, elapsed;
    double endFwProj, startBkProj, endBkProj, elapsedFwProj, elapsedBkProj;

    // Allocate temporary vectors
    // FIXME image vector missing
    float *fwproj = MALLOC(float, nRows);
    float *correlation = MALLOC(float, nRows);
    float *update = MALLOC(float, nColumns);
    float *norm = MALLOC(float, nColumns);

    for (i = 0; i < nRows; ++i) fwproj[i] = correlation[i] = 0.0;
    for (i = 0; i < nColumns; ++i) update[i] = norm[i] = 0.0;

    startTime = MPI_Wtime();
    calcColumnSums(mpi, norm, nColumns);
    endTime = MPI_Wtime();
    elapsed = endTime - startTime;

    #ifndef MESSUNG
    if (mpi.rank == 0) {
        std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]:"
            << "Calculated norm, elapsed time: " << elapsed << "s\n";
    }
    #endif

    // Fill image with initial estimate
    initImage(mpi, norm, image, lmsino, nColumns);

    if (mpi.rank == 0) {
        #ifndef MESSUNG
            std::cout << "MLEM [" << mpi.rank << "/" << mpi.size << "]:"
                << "Starting " << nIterations << " MLEM iterations" << std::endl;
            std::cout << "#, Elapsed Time (seconds), Forward Projection (seconds), "
                "Backward Projection (seconds), Image Sum" << std::endl;
        #else
            std::cout << "#, elapsed time (seconds)" << std::endl;
        #endif
    }

    for (int iter = 0; iter < nIterations; ++iter) {
        startTime = MPI_Wtime();
        calcFwProj(mpi, image, fwproj, nRows);
        #ifndef MESSUNG
            endFwProj = MPI_Wtime();
            calcCorrel(fwproj, lmsino, correlation, nRows);
            startBkProj = MPI_Wtime();
            calcBkProj(mpi, correlation, update, nColumns);
            endBkProj = MPI_Wtime();
        #else
            calcCorrel(fwproj, lmsino, correlation, nRows);
            calcBkProj(mpi, correlation, update, nColumns);
        #endif
        calcUpdate(update, norm, image, nColumns);
        endTime = MPI_Wtime();
        elapsed = endTime - startTime;

        #ifndef MESSUNG
            elapsedFwProj = endFwProj - startTime;
            elapsedBkProj = endBkProj - startBkProj;

            MPI_Reduce(&elapsedFwProj, &gFwProj, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&elapsedBkProj, &gBkProj, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        #endif

        // Max time between all nodes
        MPI_Reduce(&elapsed, &gElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (mpi.rank == 0) {
            #ifndef MESSUNG
                for(i = 0, sum = 0; i < image.size(); ++i) sum += image[i];

                printf("%d, %.15f, %.15f, %.15f, %.15lf\n", iter, gElapsed,
                    gFwProj, gBkProj, sum);
            #else
                printf("%d, %.15f \n", iter, gElapsed);
            #endif
        }

        // TODO reconsider
        MPI_Barrier(MPI_COMM_WORLD);
    }

    FREE(fwproj);
    FREE(correlation);
    FREE(update);
    FREE(norm);
}


#ifdef _OMP_
void splitMatrix(Csr4Matrix* matrix, MpiData& mpi)
{
    uint64_t sum = 0, numOfElem = 0, totalElements;
    uint32_t nRows = matrix->rows();
    uint32_t nColumns = matrix->columns();

    float avgElemsPerNode, avgElemsPerThread;
    int idx = 0;

    totalElements = matrix->elements();

    // Creating row index vector which starts at 0
    const uint64_t *rowidxA = matrix->getRowIdx();
    const RowElement<float> *rowElements = matrix->getData();
    uint64_t *rowidx = (uint64_t *) malloc((nRows + 1) * sizeof(uint64_t));

    rowidx[0] = 0;
    for (uint32_t row = 0; row < nRows; ++row) rowidx[row + 1] = rowidxA[row];

    // Allocate vector of vector of CSR structs, one vector per thread, one CSR per cache block
    mpi.threadsCSR = std::vector<std::vector<CSR>>(mpi.numberOfThreads, std::vector<CSR>(mpi.nCBlocks));

    // Assign a range of rows for each MPI node
    avgElemsPerNode = (float) totalElements / (float) mpi.size;
    if (idx == mpi.rank) mpi.range.startRow = 0;
    for (uint32_t row = 0; row < nRows; ++row) {
        sum += matrix->elementsInRow(row);
        numOfElem += matrix->elementsInRow(row);
        if (sum > avgElemsPerNode * (idx + 1)) {
            if (mpi.rank == idx) {
                mpi.range.endRow = row;
                mpi.numberOfElements = numOfElem;
            }
            ++idx;
            numOfElem = 0;
            if (mpi.rank == idx) mpi.range.startRow = row;
        }
    }

    if (mpi.rank == mpi.size - 1) {
        mpi.range.endRow = nRows;
        mpi.numberOfElements = numOfElem;
    }

    idx = 0;
    sum = 0;
    numOfElem = 0;

    avgElemsPerThread = (float) mpi.numberOfElements / (float) mpi.numberOfThreads;
    std::cout << "Average elements per thread " << avgElemsPerThread << std::endl;
    // Assign a range of rows for each thread of each node
    for (auto& csr : mpi.threadsCSR[idx]) csr.startRow = mpi.range.startRow;
    for (int row = mpi.range.startRow; row < mpi.range.endRow && idx < mpi.numberOfThreads - 1; ++row) {
        sum += matrix->elementsInRow(row);
        if (sum > avgElemsPerThread * (idx + 1)) {
            for (auto& csr: mpi.threadsCSR[idx]) csr.endRow = row;
            ++idx;
            for (auto& csr: mpi.threadsCSR[idx]) csr.startRow = row;
        }
    }
    for (auto& csr: mpi.threadsCSR[mpi.numberOfThreads - 1]) csr.endRow = mpi.range.endRow;

    for (int i = 0; i < mpi.numberOfThreads; ++i) {
        std::cout << "Mlem partitioning [" << mpi.rank << "/" << mpi.size << "-"
                  << i << "/" << mpi.numberOfThreads
                  << "]: range from " << mpi.threadsCSR[i][0].startRow << " to "
                  << mpi.threadsCSR[i][0].endRow << std::endl;
     }

    // Create the split matrix for each node or each thread
    #pragma omp parallel
    {
        const int tidx = omp_get_thread_num();
        auto& thread = mpi.threadsCSR[tidx];

        // Each thread runs over the matrix twice. First pass, collect number of elements in each block.
        // Then allocate arrays (on HBM, if available). Second pass, copy elements.

        uint32_t startRow = thread[0].startRow;
        uint32_t endRow = thread[0].endRow;

        for (auto& csr : thread) csr.nElements = 0;

        // 1st pass
        for (uint32_t r=startRow; r<endRow; ++r) {
            for (uint64_t c=rowidx[r]; c<rowidx[r + 1]; ++c) {
                uint32_t colidx = rowElements[c].column();
                int block = colidx * mpi.nCBlocks / nColumns;
                if (block >= mpi.nCBlocks) block = mpi.nCBlocks - 1;
                thread[block].nElements += 1;
            }
        }

        // Allocate per block
        for (auto& csr : thread) {
            csr.nRowIndex = endRow - startRow + 1;
            csr.nColumn = csr.nElements;
            // csr.startRow and csr.endRow are already set
            csr.columns = MALLOC(uint32_t, csr.nColumn);
            if (!csr.columns) throw std::runtime_error("Memory allocation failed");
            csr.values = MALLOC(float, csr.nElements);
            if (!csr.values) throw std::runtime_error("Memory allocation failed");
            csr.rowIndex = MALLOC(uint64_t, csr.nRowIndex);
            if (!csr.rowIndex) throw std::runtime_error("Memory allocation failed");
        }

        // 2nd pass, copy elements. Need to calculate new values in array rowIndex
        // Use CSR.nElements as helper counter
        for (auto& csr : thread) { csr.nElements = 0; csr.rowIndex[0] = 0; }
        for (uint32_t r=startRow; r<endRow; ++r) {
            for (uint64_t c=rowidx[r]; c<rowidx[r + 1]; ++c) {
                auto& elem = rowElements[c];
                int block = elem.column() * mpi.nCBlocks / nColumns;
                if (block >= mpi.nCBlocks) block = mpi.nCBlocks - 1;
                thread[block].columns[thread[block].nElements] = elem.column();
                thread[block].values[thread[block].nElements] = elem.value();
                thread[block].nElements += 1;
            }
            auto rr = r - startRow + 1;
            for (auto& csr : thread) csr.rowIndex[rr] = csr.nElements;
        }
    }

    free(rowidx);
    #ifndef MESSUNG
        printf("\nThreads Partitioning\n"
            "Rows: %d, Columns: %d Matrix elements: %lu, \n"
            "MPI Node: %d, avg per node: %f, avg per thread: %f\n",
            matrix->rows(), matrix->columns(), matrix->elements(),
            mpi.rank, avgElemsPerNode, avgElemsPerThread);

        uint64_t totalE = 0, totalR = 0;
        for (int i = 0; i < mpi.numberOfThreads; ++i) {
            printf("MLEM[%d [%d/%d]/ %d]: Range from %d - %d, Elements: %d\n",
                mpi.rank, i, mpi.numberOfThreads, mpi.size,
                mpi.threadsCSR[i][0].startRow, mpi.threadsCSR[i][0].endRow,
                mpi.threadsCSR[i][0].nColumn);

            totalE += mpi.threadsCSR[i][0].nColumn;
            totalR += mpi.threadsCSR[i][0].nRowIndex;
        }
        printf("Total elements: %lu, total Rows: %lu \n", totalE, totalR);
    #endif
}

#else

void splitMatrix(Csr4Matrix* matrix, MpiData& mpi)
{
    uint64_t sum = 0, numOfElem = 0, totalElements = 0;
    uint32_t row = 0;
    int idx = 0;

    totalElements = matrix->elements();

    // Creating correct row index vector; starting with 0!
    const uint64_t *rowidxA = matrix->getRowIdx();
    const RowElement<float> *rowElements = matrix->getData();
    uint64_t *rowidx = (uint64_t *) malloc((matrix->rows() + 1) * sizeof(uint64_t));

    rowidx[0] = 0;
    for (row = 0; row < matrix->rows(); ++row) rowidx[row + 1] = rowidxA[row];

    mpi.mpiCSR = MALLOC(CSR, 1);

    // Assign a range of rows for each MPI node
    float avgElemsPerNode = (float)totalElements / (float)mpi.size;
    if (0 == mpi.rank) mpi.range.startRow = 0;
    for (row = 0; row < matrix->rows(); ++row) {
        sum += matrix->elementsInRow(row);
        numOfElem += matrix->elementsInRow(row);
        if (sum > avgElemsPerNode * (idx + 1)) {
            if (mpi.rank == idx) {
                mpi.range.endRow = row;
                mpi.numberOfElements = numOfElem;
            }
            ++idx;
            numOfElem = 0;
            if (mpi.rank == idx) mpi.range.startRow = row;
        }
    }
    if (mpi.rank == mpi.size - 1) {
        mpi.range.endRow = matrix->rows();
        mpi.numberOfElements = numOfElem;
    }

    // Assgin a range of rows for each thread in MPI (OMP)
    mpi.mpiCSR->startRow = mpi.range.startRow;
    mpi.mpiCSR->endRow = mpi.range.endRow;

    // Create the the split matrix for each node or each thread
    uint32_t i, start, end, length;
    uint64_t base;

    std::vector<uint64_t> *rowIndexTemp = new std::vector<uint64_t>();
    std::vector<uint32_t> *columnsTemp = new std::vector<uint32_t>();
    std::vector<float> *valuesTemp = new std::vector<float>();

    start = mpi.mpiCSR->startRow;
    end = mpi.mpiCSR->endRow;
    base = rowidx[start];

    // Building row index array
    for (i = start; i < end; ++i) rowIndexTemp->push_back(rowidx[i] - base);
    if (mpi.rank < mpi.size - 1)
            rowIndexTemp->push_back(rowidx[end] - base);

    // Building columns & values arrays
    for (i = rowidx[start]; i < rowidx[end]; ++i) {
        columnsTemp->push_back(rowElements[i].column());
        valuesTemp->push_back(rowElements[i].value());
    }

    mpi.mpiCSR->nRowIndex = rowIndexTemp->size();
    mpi.mpiCSR->nColumn = columnsTemp->size();

    // Allocate memory
    mpi.mpiCSR->rowIndex = MALLOC(uint64_t, mpi.mpiCSR->nRowIndex);
    mpi.mpiCSR->columns = MALLOC(uint32_t, mpi.mpiCSR->nColumn);
    mpi.mpiCSR->values = MALLOC(float, mpi.mpiCSR->nColumn);

    // Copy the values from the temp arrays to the array real ones
    for (i = 0; i < mpi.mpiCSR->nRowIndex; ++i)
        mpi.mpiCSR->rowIndex[i] = rowIndexTemp->at(i);

    for (i = 0; i < mpi.mpiCSR->nColumn; ++i) {
        mpi.mpiCSR->columns[i] = columnsTemp->at(i);
        mpi.mpiCSR->values[i] = valuesTemp->at(i);
    }

    delete rowIndexTemp;
    delete columnsTemp;
    delete valuesTemp;

    free(rowidx);

    #ifndef MESSUNG
        printf("MLEM[%d/%d]: Range from %d to %d, Elements: %d\n",
            mpi.rank, mpi.size, mpi.mpiCSR->startRow, mpi.mpiCSR->endRow,
            mpi.mpiCSR->nColumn);
    #endif
}
#endif


void cleanMemory(MpiData& mpi)
{
    #ifdef _OMP_
        #pragma omp parallel
        {
            const int tidx = omp_get_thread_num();
            auto& thread = mpi.threadsCSR[tidx];

            // Deallocate per block
            for (auto& csr : thread) {
                FREE(csr.columns);
                FREE(csr.values);
                FREE(csr.rowIndex);
            }
        }
    #else
        FREE(mpi.mpiCSR->rowIndex);
        FREE(mpi.mpiCSR->columns);
        FREE(mpi.mpiCSR->values);
        FREE(mpi.mpiCSR);
    #endif
}


int main(int argc, char *argv[])
{
    ProgramOptions progops = handleCommandLine(argc, argv);

    #ifndef MESSUNG
        std::cout << "Matrix file: " << progops.mtxfilename << std::endl;
        std::cout << "Input file: "  << progops.infilename << std::endl;
        std::cout << "Output file: " << progops.outfilename << std::endl;
        std::cout << "Iterations: " << progops.iterations << std::endl;
        std::cout << "Number of threads each MPI node: " << progops.numberOfThreads << std::endl;
        std::cout << "Number of cache blocks: " << progops.nCBlocks << std::endl;
    #endif

    Csr4Matrix* matrix = new Csr4Matrix (progops.mtxfilename);
    uint32_t nRows = matrix->rows();
    uint32_t nColumns = matrix->columns();

    Vector<int> lmsino(progops.infilename);
    Vector<float> image(nColumns, 0.0);

    matrix->mapRows(0, nRows);

    MpiData mpi = initializeMpiOmp(argc, argv, progops.numberOfThreads, progops.nCBlocks);

    splitMatrix(matrix, mpi);

    std::cout << "Ended Split Matrix " << std::endl;

    //if (mpi.rank == 0) delete matrix;

    // Main algorithm
    mlem(mpi, lmsino, image, progops.iterations, nRows, nColumns);

    if (0 == mpi.rank) image.writeToFile(progops.outfilename);

    cleanMemory(mpi);
    MPI_Finalize();

    return 0;
}
