#include "../include/csr4matrix.hpp"
#include "../include/vector.hpp"

extern "C" {
#include "laik-backend-mpi.h"
#include "laik-internal.h"
}

typedef unsigned int uint;

// C++ additions to LAIK header
inline Laik_DataFlow operator|(Laik_DataFlow a, Laik_DataFlow b)
{
    return static_cast<Laik_DataFlow>(static_cast<int>(a) | static_cast<int>(b));
}

#include <iostream>
#include <string>
#include <stdexcept>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#ifdef MESSUNG 
#include <stdio.h>
#endif


struct ProgramOptions
{
    std::string mtxfilename;
    std::string infilename;
    std::string outfilename;
    int iterations;
    int checkpointing;
};



struct Range
{
    int start;
    int end;
};


struct SubRow
{
    int row;
    int offset;
};


struct SubRowSlice
{
    SubRow from;
    SubRow to;
};


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


#if 0
// For element-wise weighted partitioning: number of elems in row
double getEW(Laik_Index* i, const void* d)
{
    // SpM* m = (SpM*) d;
    const Csr4Matrix* m = static_cast<const Csr4Matrix*>(d);
    int ii = i->i[0];  // first dimension of Laik_Index

    // Return (float) (m->row[ii + 1] - m->row[ii]);
    return m->elementsInRow(ii);
}
#endif

struct RowData {
    // local row index vector
    std::vector<uint64_t> rowIdx;

    // global row numbers
    size_t fromRow;
    size_t toRow;
};

struct Slice {
    int64_t from;
    int64_t to;
};

bool operator==(const Slice& lhs, const Slice& rhs) {
    return lhs.from == rhs.from && lhs.to == rhs.to;
}

struct Hash
{
    std::size_t operator()(const Slice& s) const noexcept
    {
        std::size_t h1 = std::hash<uint64_t>{}(s.from);
        std::size_t h2 = std::hash<uint64_t>{}(s.to);
        return h1 ^ (h2 < 1);
    }
};

std::unordered_map<Slice, RowData, Hash> cache;


RowData cache_get(const Csr4Matrix& matrix, const Slice& slice)
{
    auto search = cache.find(slice);
    if (search != cache.end()) return search->second;

    const uint32_t rowCount = matrix.rows();
    const uint64_t* origRowIdx = matrix.getRowIdx();
    
    laik_log(2, "origRowIdx: %d %d %d %d %d", origRowIdx[0], origRowIdx[1], origRowIdx[2], origRowIdx[3], origRowIdx[4]);
    
    auto it = std::upper_bound(origRowIdx, origRowIdx + rowCount, slice.from);
    
    
    
    if (std::distance(origRowIdx, it) == 0) {
        // Original row index vector is missing first element "0"
        size_t fromRow = 0;
        size_t toRow = std::distance(origRowIdx,
            std::lower_bound(origRowIdx, origRowIdx + rowCount, slice.to - 1));
        size_t len = toRow - fromRow + 2;
        RowData rv = { std::vector<uint64_t>(len), fromRow, toRow };
        std::copy(origRowIdx, origRowIdx + len - 1, std::begin(rv.rowIdx) + 1);
        rv.rowIdx[0] = slice.from;
        rv.rowIdx.back() = slice.to;
        cache[slice] = rv;
        return rv;
    } // else

    size_t fromRow = std::distance(origRowIdx, it);
    size_t toRow = std::distance(origRowIdx,
        std::upper_bound(origRowIdx, origRowIdx + rowCount, slice.to - 1));
    size_t len = toRow - fromRow + 1;  // number of rows + 1
    
    laik_log(2, "origRowIdx: %p, it: %p", origRowIdx, it -1);
   
    laik_log(2, "Within %s: fromRow: %d, toRow %d, len %d", __FUNCTION__, fromRow, toRow, len);
    RowData rv { std::vector<uint64_t>(len), fromRow, toRow };
    std::copy(origRowIdx + fromRow, origRowIdx + fromRow + len - 1, std::begin(rv.rowIdx) + 1);
    rv.rowIdx[0] = slice.from;
    rv.rowIdx.back() = slice.to;
    cache[slice] = rv;
    return rv;
}


void calcColumnSums(Laik_Group* world,
                    Laik_Partitioning* p,
                    const Csr4Matrix& matrix, Laik_Data* norm)
{
    //laik_switchto_partitioning(norm, pAll, LAIK_DF_Init, LAIK_RO_Sum);
      laik_switchto_flow(norm, LAIK_DF_Init, LAIK_RO_Sum);

    
    float* res;
    laik_map_def1(norm, (void**) &res, 0);

    // Loop over all local slices

    for (int sNo = 0; ; sNo++) {
        Slice slice;
        Laik_TaskSlice* slc = laik_my_slice_1d(p, sNo, &slice.from, &slice.to);
        if (slc == 0) break;

        RowData rv = cache_get(matrix, slice);
        auto fromRow = rv.fromRow;
        auto toRow = rv.toRow;
        matrix.mapRows(fromRow, toRow - fromRow + 1);

        for (size_t r = 0; r < rv.rowIdx.size() - 1; r++) {
            std::for_each(matrix.beginRow(r, rv.rowIdx), matrix.endRow(r, rv.rowIdx),
                [&](const RowElement<float>& e){ res[e.column()] += e.value();
            });
        }

        float s = 0.0;
        for (uint i = 0; i < matrix.columns(); i++) s += res[i];
#ifndef MESSUNG
        laik_log(LAIK_LL_Info, "Range %d - %d: Sum: %lf \n", fromRow, toRow, s);
#endif
    }

     laik_switchto_flow(norm, LAIK_DF_Preserve, LAIK_RO_Sum);

    laik_map_def1(norm, (void**) &res, 0);

#ifndef MESSUNG
    float s = 0.0;
    for(uint i = 0; i < matrix.columns(); i++) s += res[i];
    laik_log(LAIK_LL_Info, "Norm_Sum: %lf", s);
#endif
}


void initImage(Laik_Data* norm, Vector<float>& image, const Vector<int>& lmsino)
{
    float* n;
    laik_map_def1(norm, (void**) &n, 0);

    // Sum up norm vector, creating sum of all matrix elements
    float sumnorm = 0.0;
    for (size_t i=0; i<image.size(); ++i) sumnorm += n[i];
    float sumin = 0.0;
    for (size_t i=0; i<lmsino.size(); ++i) sumin += lmsino[i];

    float initial = static_cast<float>(sumin / sumnorm);

#ifndef MESSUNG
    laik_log(LAIK_LL_Info, "Init: SumNorm= %lf, sumin= %lf, initial value = %lf\n", sumnorm, sumin, initial);
#endif
    for (size_t i=0; i<image.size(); ++i) image[i] = initial;
}


void calcFwProj(Laik_Group* world,
                Laik_Partitioning* p,
                const Csr4Matrix& matrix,
                const Vector<float>& input,
                Laik_Data* result,
                std::chrono::duration<float>& comp_time,
                std::chrono::duration<float>& total_time
                )
{
    //laik_switchto_new_phase(result, world, laik_All,LAIK_DF_Init | LAIK_DF_ReduceOut | LAIK_DF_Sum);
    //laik_switchto_partitioning(result, pAll, LAIK_DF_Init, LAIK_RO_Sum);
     laik_switchto_flow(result, LAIK_DF_Init, LAIK_RO_Sum);
    
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> s_comp, s_comp_end, fin;

    float* res;
    laik_map_def1(result, (void**) &res, 0);

    // Loop over all local slices
    for(int sNo = 0; ; sNo++) {

        //Slice slice;
        //Laik_TaskSlice* slc = laik_my_slice_1d(p, sNo, &slice.from, &slice.to);
        //Laik_TaskSlice* slc = laik_phase_myslice_1d(ap, sNo, &slice.from, &slice.to);
        
        Slice slice;
        Laik_TaskSlice* slc = laik_my_slice_1d(p, sNo, &slice.from, &slice.to);
        
        laik_log(2, "My Slice: from %d, to %d", slice.from, slice.to);


        if (slc == 0) break;

        RowData rv = cache_get(matrix, slice);
        auto fromRow = rv.fromRow;
        auto toRow = rv.toRow;
        matrix.mapRows(fromRow, toRow - fromRow + 1);
        laik_log(2, "My Slice: from %d, to %d", fromRow, toRow);


        s_comp = std::chrono::system_clock::now();

        for(size_t r = 0; r < rv.rowIdx.size() - 1; r++) {
            std::for_each(matrix.beginRow(r, rv.rowIdx), matrix.endRow(r, rv.rowIdx),
                [&](const RowElement<float>& e){ res[r + fromRow] += (float)e.value() * input[e.column()]; });
            if (r<100 ) laik_log(2, "index: %d, value:%f", r+fromRow, res[r+fromRow]);
        }

        s_comp_end = std::chrono::system_clock::now();
        comp_time += (s_comp_end - s_comp);
    }

    //laik_switchto_new_phase(result, world, laik_All, LAIK_DF_CopyIn);
    //laik_switchto_partitioning(result, pAll, LAIK_DF_Preserve, LAIK_RO_Sum);
         laik_switchto_flow(result, LAIK_DF_Preserve, LAIK_RO_Sum);



    for(int sNo = 0; ; sNo++) {

        //Slice slice;
        //Laik_TaskSlice* slc = laik_my_slice_1d(p, sNo, &slice.from, &slice.to);
        //Laik_TaskSlice* slc = laik_phase_myslice_1d(ap, sNo, &slice.from, &slice.to);
        
        Slice slice;
        Laik_TaskSlice* slc = laik_my_slice_1d(p, sNo, &slice.from, &slice.to);
        
        if (slc == 0) break;

        RowData rv = cache_get(matrix, slice);
        auto fromRow = rv.fromRow;
        auto toRow = rv.toRow;
        matrix.mapRows(fromRow, toRow - fromRow + 1);

        laik_map_def1(result, (void**) &res, 0);


        for(size_t r = 0; r < rv.rowIdx.size() - 1; r++) {
            if (r<100 ) laik_log(2, "index: %d, value:%f", r, res[r+fromRow]);
        }
    }

        
    
    fin = std::chrono::system_clock::now();
    total_time += (fin - start);
}


void calcCorrel(
        Laik_Data* fwproj,
        const Vector<int>& input,
        Vector<float>& correlation,
        std::chrono::duration<float>& comp_time,
        std::chrono::duration<float>& total_time
        )
{
    float* fwp;
    uint64_t size;
    
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> start2, fin;

    laik_map_def1(fwproj, (void**) &fwp, &size);
    
    start2 = std::chrono::system_clock::now();
    for (size_t i=0; i<size; ++i)
        correlation[i] = (fwp[i] != 0.0) ? (input[i] / fwp[i]) : 0.0;

    fin = std::chrono::system_clock::now();
    total_time += (fin - start2);
    comp_time += (fin - start);
}


void calcBkProj(Laik_Group* world,
                Laik_Partitioning* p, 
                const Csr4Matrix& matrix,
                const Vector<float>& correlation,
                Laik_Data* update,
                std::chrono::duration<float>& comp_time,
                std::chrono::duration<float>& total_time)
{
    //laik_switchto_new_phase(update, world, laik_All,
     //                 LAIK_DF_Init | LAIK_DF_ReduceOut | LAIK_DF_Sum);
    
    //laik_switchto_partitioning(update, pAll, LAIK_DF_Init, LAIK_RO_Sum);
     laik_switchto_flow(update, LAIK_DF_Init, LAIK_RO_Sum);

    
    float* res;
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> s_comp, s_comp_end,fin;

    laik_map_def1(update, (void**) &res, 0);

    // Loop over all local slices
    for(int sNo = 0; ; sNo++) {

        //Slice slice;
        //Laik_TaskSlice* slc = laik_phase_myslice_1d(ap, sNo, &slice.from, &slice.to);
        
        Slice slice;
        Laik_TaskSlice* slc = laik_my_slice_1d(p, sNo, &slice.from, &slice.to);
        
        if (slc == 0) break;

        RowData rv = cache_get(matrix, slice);
        auto fromRow = rv.fromRow;
        auto toRow = rv.toRow;
        matrix.mapRows(fromRow, toRow - fromRow + 1);

        s_comp = std::chrono::system_clock::now();

        for (size_t r = 0; r < rv.rowIdx.size() - 1; r++) {
            std::for_each(matrix.beginRow(r, rv.rowIdx), matrix.endRow(r, rv.rowIdx),
                [&](const RowElement<float>& e){ res[e.column()] += (float)e.value() * correlation[r + fromRow]; });
        }

        s_comp_end = std::chrono::system_clock::now();
        comp_time += (s_comp_end - s_comp);
    }


    //laik_switchto_partitioning(update, pAll, LAIK_DF_Preserve, LAIK_RO_Sum);
         laik_switchto_flow(update, LAIK_DF_Preserve, LAIK_RO_Sum);

    fin = std::chrono::system_clock::now();
    total_time += (fin - start);
}


void calcUpdate(
        Laik_Data* update,
        Laik_Data* norm,
        Vector<float>& image,
        std::chrono::duration<float>& comp_time,
        std::chrono::duration<float>& total_time
        )
{
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> start2, fin;
    float* up;
    laik_map_def1(update, (void**) &up, 0);
    float* nm;
    laik_map_def1(norm, (void**) &nm, 0);

    start2 = std::chrono::system_clock::now();
    for (size_t i=0; i<image.size(); ++i)
        image[i] *= (nm[i] != 0.0) ? (up[i] / nm[i]) : up[i];

    fin = std::chrono::system_clock::now();
    total_time += (fin - start2);
    comp_time += (fin - start);
}


void mlem(Laik_Instance* inst, Laik_Group* world, Laik_Partitioning* p,
          const Csr4Matrix& matrix, const Vector<int>& lmsino,
          Vector<float>& image, int nIterations)
{
    uint32_t nRows = matrix.rows();
    uint32_t nColumns = matrix.columns();
    std::chrono::duration<float> compute_time, total_time;

    // Allocate temporary vectors
    Laik_Data* fwproj = laik_new_data_1d(inst, laik_Float, nRows);
    laik_data_set_name (fwproj, "fwProj");
    laik_switchto_new_partitioning(fwproj, world, laik_All, LAIK_DF_None, LAIK_RO_None);
    
    Vector<float> correlation(nRows, 0.0); // could update vector fwproj instead
    //Vector<float> update(nColumns, 0.0);
    Laik_Data* update = laik_new_data_1d(inst, laik_Float, nColumns);
    laik_data_set_name(update, "update");
    laik_switchto_new_partitioning(update, world, laik_All, LAIK_DF_None, LAIK_RO_None);


    // Calculate column sums ("norm")
    Laik_Data* norm = laik_new_data_1d(inst, laik_Float, nColumns);
    laik_data_set_name(norm, "norm");
    laik_switchto_new_partitioning(norm, world, laik_All, LAIK_DF_None, LAIK_RO_None);

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    calcColumnSums(world, p, matrix, norm);
    end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;

#ifndef MESSUNG
    laik_log(LAIK_LL_Info, "Calculated norm, elapsed time: %f\n", elapsed_seconds.count());
#endif

    // Fill image with initial estimate
    initImage(norm, image, lmsino);

#ifndef MESSUNG
    laik_log(LAIK_LL_Info, "Starting %d MLEM Iterations\n", nIterations);
#endif
    for (int iter=0; iter<nIterations; ++iter) {

        compute_time = std::chrono::duration<float>::zero();
        total_time = std::chrono::duration<float>::zero();
        
        calcFwProj(world, p, matrix, image, fwproj, compute_time, total_time);
        calcCorrel(fwproj, lmsino, correlation, compute_time, total_time);
        calcBkProj(world, p, matrix, correlation, update, compute_time, total_time);
        calcUpdate(update, norm, image, compute_time, total_time);

        // Debug: sum over image values
        float s = 0.0;
        for(uint i = 0; i < matrix.columns(); i++) s += image[i];
#ifndef MESSUNG
        laik_log(LAIK_LL_Info, "Finished Iteration %d, Time: %f(%f/%f/%f)\n",
                 iter + 1,
                 total_time.count(),
                 compute_time.count(),
                 laik_get_total_time() - laik_get_backend_time(),
                 laik_get_backend_time()
                 );
#else
        printf("%d, %lf, %lf, %lf, %lf\n",
               iter + 1,
               total_time.count(),
               compute_time.count(),
               laik_get_total_time() - laik_get_backend_time(),
               laik_get_backend_time() );
#endif

        laik_reset_profiling(inst);
#ifndef MESSUNG
        laik_log(LAIK_LL_Info, "Image Sum: %f\n", s);
#endif
    }
}


void mlemPartitioner(Laik_Partitioner* pr,
                     Laik_Partitioning* p, Laik_Partitioning* oldP)
{
    // Laik_Space* space = ba->space;  // unused
    Laik_Group* g = p->group;
    Csr4Matrix* mtx = static_cast<Csr4Matrix*>(pr->data);

    int sliceCount = g->size;
    uint64_t elementsPerSlice = (mtx->elements() + sliceCount - 1) / sliceCount;
    uint64_t elementCount = 0;
    uint64_t task = 0;

    SubRowSlice* slice_data = new SubRowSlice[sliceCount];  // FIXME memory leak

    Laik_Slice sl;
    sl.from = { 0, 0, 0 }; // Laik_Index
    sl.to = { 0, 0, 0, };

    int startRow = 0;
    int startOffset = 0;

    for (uint32_t row = 0; row < mtx->rows(); ++row) {
        elementCount += mtx->elementsInRow(row);
        if (elementCount >= sl.from.i[0] + elementsPerSlice) {
            sl.to.i[0] = std::min(sl.from.i[0] + elementsPerSlice, mtx->elements());

            uint32_t offset = sl.from.i[0] + elementsPerSlice - elementCount + mtx->elementsInRow(row);

            slice_data[task].from.row = startRow;
            slice_data[task].from.offset = startOffset;
            slice_data[task].to.row = row;
            slice_data[task].to.offset = offset;

            if (offset == mtx->elementsInRow(row)) {
                slice_data[task].to.row = row + 1;
                slice_data[task].to.offset = 0;
            }

            laik_append_slice(p, task, &sl, 0, (void*)&slice_data[task]);
            ++task;
            sl.from = sl.to;
            startRow = row;
            startOffset = offset;
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
#endif
    Csr4Matrix matrix(progops.mtxfilename);
    //Csr4Matrix matrix = Csr4Matrix::testMatrix();

#ifndef MESSUNG
    std::cout << "Matrix rows (LORs): " << matrix.rows() << std::endl;
    std::cout << "Matrix cols (VOXs): " << matrix.columns() << std::endl;
#endif
    Vector<int> lmsino(progops.infilename);
    Vector<float> image(matrix.columns(), 0.0);

    Laik_Instance* inst = laik_init_mpi(&argc, &argv);
    Laik_Group* world = laik_world(inst);

    laik_enable_profiling(inst);

    // 1d space, block partitioning of matrix elements
    Laik_Space* space = laik_new_space_1d(inst, matrix.elements());
    Laik_Partitioner* part = laik_new_block_partitioner1();
    Laik_Partitioning* p = laik_new_partitioning(part, world, space, 0);
    //Laik_AccessPhase* ap = laik_new_accessphase(world, space, part, nullptr);

    mlem(inst, world, p, matrix, lmsino, image, progops.iterations);

    if (laik_myid(world) == 0)
        image.writeToFile(progops.outfilename);

    laik_finalize(inst);

    return 0;
}
