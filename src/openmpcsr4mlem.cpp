/**
    Copyright © 2017 Technische Universitaet Muenchen
    Authors: Rami Al Rihawi
            Tilman Kuestner
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

#include "../include/csr4matrix.hpp"
#include "../include/vector.hpp"

#include <omp.h>

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cassert>
#include <algorithm>
#include <chrono>

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
};


// Global declaration 
Csr4Matrix* matrix;

#ifdef _PINNING_ 
    int numberOfThreads = 0;
    uint64_t totalElements = 0;

    struct CSR
    {
        int nRowIndex, nColumn;
        uint32_t startRow, endRow; 
        uint32_t *columns;
        uint64_t *rowIndex;
        float *values; 
    };

    CSR** threadsCSR;
#endif


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


void calcColumnSums(float* norm)
{
    #ifdef _PINNING_
        for (size_t i = 0; i < numberOfThreads; ++i) {
            for (size_t j = 0; j < threadsCSR[i]->nColumn; ++j) {
                norm[threadsCSR[i]->columns[j]] += threadsCSR[i]->values[j];
            }
        }
    #else 
        for (uint32_t row = 0; row < matrix->rows(); ++row) {
            std::for_each(matrix->beginRow2(row), matrix->endRow2(row),
                [&](const RowElement<float>& e) { 
                    norm[e.column()] += e.value();
                });
        }
    #endif
}


void initImage(const float* norm, Vector<float>& image, const Vector<int>& lmsino, 
    size_t normSize)
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
        std::cout << "INIT: sumnorm=" << sumnorm << ", sumin=" << sumin
            << ", initial value=" << initial << std::endl;
    #endif
}


void calcFwProj(const Vector<float>& input, float* fwproj)
{
    #ifdef _PINNING_ 
        #pragma omp parallel 
        {
            const int tidx = omp_get_thread_num();
            float res = 0.0;
            
            uint32_t i, j, startColumnIndex, endColumnIndex; 
            uint32_t length = threadsCSR[tidx]->nRowIndex, row = threadsCSR[tidx]->startRow;

            for(i = 1, startColumnIndex = 0; 
                row < threadsCSR[tidx]->endRow && i < length; 
                ++i, ++row, startColumnIndex = endColumnIndex) {
                endColumnIndex = startColumnIndex + 
                    (threadsCSR[tidx]->rowIndex[i] - threadsCSR[tidx]->rowIndex[i-1]);
                res = 0.0;

                for(j = startColumnIndex; j < endColumnIndex; ++j) 
                    res += threadsCSR[tidx]->values[j] * input[threadsCSR[tidx]->columns[j]];
                
                fwproj[row] = res;
            }
        }
    #else 
        #pragma omp parallel for schedule(dynamic)
        for (uint32_t row = 0; row < matrix->rows(); ++row) {
            float res = 0.0;
        
            std::for_each(matrix->beginRow2(row), matrix->endRow2(row),
                [&](const RowElement<float>& e) { 
                    res += e.value() * input[e.column()]; 
                    });

            fwproj[row] = res;

            /*
            auto start = matrix.beginRow2(row);
            auto end = matrix.endRow2(row);
            for (auto it=start; it!=end; ++it) {
                RowElement<float> e = *it;
                res += e.value() * input[e.column()];
            }
            */
        }
    #endif
}


void calcCorrel(float* fwproj, const Vector<int>& input, float* correlation, 
    size_t fwprojSize)
{
    for (size_t i = 0; i < fwprojSize; ++i)
        correlation[i] = (fwproj[i] != 0.0) ? (input[i] / fwproj[i]) : 0.0;
}

/* Version 1: require gcc 6.1, brings OpenMP 4.5
   #pragma omp parallel for reduction(+:udpate.prt()[:udpate.size()])
*/

// Version 2, critical section
void calcBkProj(float* correlation, float* update, size_t updateSize)
{
    for (size_t i = 0; i < updateSize; ++i) update[i] = 0.0;

    #pragma omp parallel
    {
        Vector<float> private_update(updateSize, 0);

        #ifdef _PINNING_ 
            const int tidx = omp_get_thread_num();

            uint32_t i, j, startColumnIndex, endColumnIndex; 
            uint32_t length = threadsCSR[tidx]->nRowIndex; 
            uint32_t row = threadsCSR[tidx]->startRow;

            for(i = 1, startColumnIndex = 0; 
                row < threadsCSR[tidx]->endRow && i < length; 
                ++i, ++row, startColumnIndex = endColumnIndex) {
                endColumnIndex = startColumnIndex + 
                    (threadsCSR[tidx]->rowIndex[i] - threadsCSR[tidx]->rowIndex[i-1]);

                for(j = startColumnIndex; j < endColumnIndex; ++j) 
                    private_update[threadsCSR[tidx]->columns[j]] += 
                        threadsCSR[tidx]->values[j] * correlation[row];
            }
        #else 
            #pragma omp for schedule(dynamic)
            for (uint32_t row = 0; row < matrix->rows(); ++row) {
                std::for_each(matrix->beginRow2(row), matrix->endRow2(row),
                 [&](const RowElement<float>& e) { 
                    private_update[e.column()] += e.value() * correlation[row]; 
                });
            }
        #endif

        #pragma omp critical
        {
            for (size_t i = 0; i < updateSize; ++i) update[i] += private_update[i];
        }
    }
}

// Version 2, avoid critical section, perform reduction by hand, may have cache issues
/*
void calcBkProj3(const Csr4Matrix& matrix, const Vector<float>& correlation, Vector<float>& update)
{
    assert(matrix.rows() == correlation.size() && matrix.columns() == update.size());

    auto length = update.size();
    std::fill(update.ptr(), update.ptr() + length, 0);

    std::vector<float> proxy_update;

    #pragma omp parallel
    {
        const int nthreads = omp_get_num_threads();
        const int tidx=omp_get_num_threads();

        #pragma omp single
        proxy_update.resize(nthreads * length, 0);

        #pragma omp for
        for (uint32_t row=0; row<matrix.rows(); ++row) {
            std::for_each(matrix.beginRow2(row), matrix.endRow2(row),
                [&](const RowElement<float>& e) {
                    proxy_update[tidx * length + e.column()] += e.value() * correlation[row];
            });
        }

        #pragma omp for
        for (size_t i = 0; i < length; ++i) {
            for (int t = 0; t < nthreads; ++t) {
                update[i] += proxy_update[t * length + i];
            }
        }
    }
}
*/


void calcUpdate(float* update, float* norm, Vector<float>& image, 
    size_t updateSize)
{
    for (size_t i = 0; i < updateSize; ++i)
        image[i] *= (norm[i] != 0.0) ? (update[i] / norm[i]) : update[i];
}


void mlem(const Vector<int>& lmsino, Vector<float>& image, int nIterations,
    uint32_t nRows, uint32_t nColumns)
{
    size_t i = 0;

    // Allocate temporary vectors
    float *fwproj = MALLOC(float, nRows);
    float *correlation = MALLOC(float, nRows);
    float *update = MALLOC(float, nColumns);
    float *norm = MALLOC(float, nColumns);

    for (i = 0; i < nRows; ++i) fwproj[i] = correlation[i] = 0.0;
    for (i = 0; i < nColumns; ++i) update[i] = norm[i] = 0.0;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed;
    
    start = std::chrono::system_clock::now();
    calcColumnSums(norm);
    end = std::chrono::system_clock::now();
    elapsed = end - start;
    
    std::cout << "Calculated norm, elapsed time: " << elapsed.count() << "s\n";

    // Fill image with initial estimate
    initImage(norm, image, lmsino, nColumns);

    std::cout << "Starting " << nIterations << " MLEM iterations" << std::endl;

    #ifndef MESSUNG
        std::chrono::time_point<std::chrono::system_clock> endFwProj, startBkProj, endBkProj;
        std::chrono::duration<double> elapsedFwProj, elapsedBkProj;

        printf("#, Elapsed Time (seconds), Forward Projection (seconds), "
            "Backward Projection (seconds), Image Sum\n");
    #else 
        printf("#, Elapsed Time (seconds)\n");
    #endif

    for (int iter=0; iter<nIterations; ++iter) {
        start = std::chrono::system_clock::now();
        calcFwProj(image, fwproj);
        #ifndef MESSUNG
            endFwProj = std::chrono::system_clock::now();
            
            calcCorrel(fwproj, lmsino, correlation, nRows);
            
            startBkProj = std::chrono::system_clock::now();
            calcBkProj(correlation, update, nColumns);
            endBkProj = std::chrono::system_clock::now();
        #else 
            calcCorrel(fwproj, lmsino, correlation, nRows);
            calcBkProj(correlation, update, nColumns);
        #endif
        calcUpdate(update, norm, image, nColumns);
        end = std::chrono::system_clock::now();

        elapsed = end - start;
        
        #ifndef MESSUNG
            elapsedFwProj = endFwProj - start; // start = startFwProj
            elapsedBkProj = endBkProj - startBkProj;

            double sum = 0;
            for(i = 0; i < image.size(); ++i) sum += image[i];

            printf("%d, %.15f, %.15f, %.15f, %.15lf\n", iter, elapsed.count(),
                elapsedFwProj.count(), elapsedBkProj.count(), sum); 
        #else 
            printf("%d, %.15f\n", iter, elapsed.count()); 
        #endif
    }
    
    FREE(fwproj);
    FREE(correlation);
    FREE(update);
    FREE(norm);
}

#ifdef _PINNING_
void splitMatrix() 
{
    uint64_t sum = 0;
    uint32_t row = 0;
    int idx = 0;

    totalElements = matrix->elements();

    // creating correct row index vector; starting with 0!
    const uint64_t *rowidxA = matrix->getRowIdx();
    const RowElement<float> *rowElements = matrix->getData();
    uint64_t *rowidx = (uint64_t *) malloc((matrix->rows() + 1) * sizeof(uint64_t));
    
    rowidx[0] = 0;  
    for (row = 0; row < matrix->rows(); ++row) rowidx[row + 1] = rowidxA[row];


    // Thread CSR preparation space allocation 
    #pragma omp parallel 
    {
        const int tidx = omp_get_thread_num();
        
        // Allocate a shared array of pointer for CSR struct of each thread
        #pragma omp master 
        {
            numberOfThreads = omp_get_num_threads();
            threadsCSR = MALLOC(CSR*, numberOfThreads);   
        }
        #pragma omp barrier 
        
        // Each thread malloc its own struct 
        threadsCSR[tidx] = MALLOC(CSR, 1);
    }

    // Assign a range of rows for each thread
    float avgElemsPerRank = (float)totalElements / (float)numberOfThreads;
    threadsCSR[idx]->startRow = 0;
    for (uint32_t row = 0; row < matrix->rows(); ++row) {
        sum += matrix->elementsInRow(row);
        if (sum > avgElemsPerRank * (idx + 1)) {
            threadsCSR[idx]->endRow = row;
            ++idx;
            threadsCSR[idx]->startRow = row;
        }
    }
    threadsCSR[numberOfThreads - 1]->endRow = matrix->rows();

    // Build ThreadCSR 
    #pragma omp parallel 
    {
        const int tidx = omp_get_thread_num();
        uint32_t i, start, end, length;
        uint64_t base;

        start = threadsCSR[tidx]->startRow;
        end = threadsCSR[tidx]->endRow;
        base = rowidx[start];

        std::vector<uint64_t> *rowIndexTemp = new std::vector<uint64_t>();
        std::vector<uint32_t> *columnsTemp = new std::vector<uint32_t>();
        std::vector<float> *valuesTemp = new std::vector<float>();
        
        // building row index array
        for (i = start; i < end; ++i) rowIndexTemp->push_back(rowidx[i] - base);
        if (tidx < numberOfThreads - 1) 
            rowIndexTemp->push_back(rowidx[threadsCSR[tidx]->endRow] - base);

        // building columns & values arrays
        for (i = rowidx[start]; i < rowidx[end]; ++i) {
            columnsTemp->push_back(rowElements[i].column());
            valuesTemp->push_back(rowElements[i].value()); 
        }
        
        // Allocate memory 
        threadsCSR[tidx]->nRowIndex = rowIndexTemp->size();
        threadsCSR[tidx]->nColumn = columnsTemp->size();
        
        // Allocate memory 
        threadsCSR[tidx]->rowIndex = 
            MALLOC(uint64_t, threadsCSR[tidx]->nRowIndex);
        threadsCSR[tidx]->columns = 
            MALLOC(uint32_t, threadsCSR[tidx]->nColumn);
        threadsCSR[tidx]->values = 
            MALLOC(float, threadsCSR[tidx]->nColumn);

        // Copy the values from the temp arrays to the array real ones
        for (i = 0; i < threadsCSR[tidx]->nRowIndex; ++i) 
            threadsCSR[tidx]->rowIndex[i] = rowIndexTemp->at(i);

        for (i = 0; i < threadsCSR[tidx]->nColumn; ++i) {
            threadsCSR[tidx]->columns[i] = columnsTemp->at(i);
            threadsCSR[tidx]->values[i] = valuesTemp->at(i);
        }

        delete rowIndexTemp;
        delete columnsTemp;
        delete valuesTemp;

        free(rowidx);
    }

    #ifndef MESSUNG 
        printf("\nThreads Partitioning\n"
            "Rows: %d, Columns: %d, \n"
            "Matrix elements: %lu, average elements per rank: %f\n", 
            matrix->rows(), matrix->columns(), matrix->elements(), 
            avgElemsPerRank);

        uint64_t totalE = 0, totalR = 0; 
        for (int i = 0; i < numberOfThreads; ++i) {
            printf("MLEM [%d/%d]: Range from %d to %d, Elements: %d\n", i, 
             numberOfThreads, threadsCSR[i]->startRow, threadsCSR[i]->endRow, 
             threadsCSR[i]->nColumn);
             totalE += threadsCSR[i]->nColumn;
             totalR += threadsCSR[i]->nRowIndex;
        }
        printf("Total elements: %lu, total Rows: %lu \n", totalE, totalR);
    #endif
}


void cleanThreadMemory() 
{
    #pragma omp parallel 
    {
        const int tidx = omp_get_thread_num();
        FREE(threadsCSR[tidx]->rowIndex);
        FREE(threadsCSR[tidx]->columns);
        FREE(threadsCSR[tidx]->values);
        FREE(threadsCSR[tidx]);

        #pragma omp master 
        FREE(threadsCSR);
    }
}
#endif


int main(int argc, char *argv[])
{
    ProgramOptions progops = handleCommandLine(argc, argv);
    
    #ifndef MESSUNG
        std::cout << "Matrix file: " << progops.mtxfilename << std::endl;
        std::cout << "Input file: "  << progops.infilename << std::endl;
        std::cout << "Output file: " << progops.outfilename << std::endl;
        std::cout << "Iterations: " << progops.iterations << std::endl;
    #endif

    matrix = new Csr4Matrix(progops.mtxfilename);
    uint32_t nRows = matrix->rows(), nColumns = matrix->columns();

    Vector<int> lmsino(progops.infilename);
    Vector<float> image(nColumns, 0.0);

    matrix->mapRows(0, nRows);

    #ifdef _PINNING_
        splitMatrix();
        delete matrix;
    #endif 

    // Main algorithm 
    mlem(lmsino, image, progops.iterations, nRows, nColumns);

    // free memory
    #ifdef _PINNING_
        cleanThreadMemory();
    #else 
        delete matrix;
    #endif 

    image.writeToFile(progops.outfilename);

    return 0;
}