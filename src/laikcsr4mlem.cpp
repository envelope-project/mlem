#include "../include/csr4matrix.hpp"
#include "../include/vector.hpp"

extern "C" {
#include "laik-backend-mpi.h"
}

// C++ additions to LAIK header (TODO: C++ interface)
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

#include <cstdlib>

typedef unsigned int uint;

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
    
    laik_log(LAIK_LL_Debug, "origRowIdx: %lu %lu %lu %lu %lu", origRowIdx[0], origRowIdx[1], origRowIdx[2], origRowIdx[3], origRowIdx[4]);
    
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
    
    laik_log(LAIK_LL_Debug, "origRowIdx: %p, it: %p", origRowIdx, it -1);
   
    laik_log(LAIK_LL_Debug, "Within %s: fromRow: %lu, toRow %lu, len %lu", __FUNCTION__, fromRow, toRow, len);
    RowData rv { std::vector<uint64_t>(len), fromRow, toRow };
    std::copy(origRowIdx + fromRow, origRowIdx + fromRow + len - 1, std::begin(rv.rowIdx) + 1);
    rv.rowIdx[0] = slice.from;
    rv.rowIdx.back() = slice.to;
    cache[slice] = rv;
    return rv;
}

/*
// For element-wise weighted partitioning: number of elems in row
double getEW(Laik_Index* i, const void* d)
{
    //SpM* m = (SpM*) d;
    const Csr4Matrix* m = static_cast<const Csr4Matrix*>(d);
    int ii = i->i[0];  // first dimension of Laik_Index

    // Return (float) (m->row[ii + 1] - m->row[ii]);
    return m->elementsInRow(ii);
}
*/

void calcColumnSums(Laik_Partitioning* p,
                    const Csr4Matrix& matrix, Laik_Data* norm)
{
    // laik_switchto_flow(norm, LAIK_DF_Init | LAIK_DF_ReduceOut | LAIK_DF_Sum);
    laik_switchto_flow(norm, LAIK_DF_Init, LAIK_RO_Sum);

    float* res;
    laik_map_def1(norm, (void**) &res, 0);

    // Loop over all local slices
    for(int sNo = 0; ; sNo++) {
        //Laik_Slice* slc = laik_my_slice(p, sNo);
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
        laik_log(LAIK_LL_Debug, "Range %lu - %lu: Sum: %lf \n", fromRow, toRow, s);
#endif
    }

    // Copyout is important to preserve data over later repartitionings
    //laik_switchto_flow(norm, LAIK_DF_CopyIn | LAIK_DF_CopyOut);
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


void calcFwProj(Laik_Partitioning* p,
                const Csr4Matrix& matrix,
                const Vector<float>& input,
                Laik_Data* result,
                std::chrono::duration<float>& comp_time,
                std::chrono::duration<float>& total_time
                )
{
    //laik_switchto_flow(result, LAIK_DF_Init | LAIK_DF_ReduceOut | LAIK_DF_Sum);
    laik_switchto_flow(result, LAIK_DF_Init, LAIK_RO_Sum);

    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> s_comp, s_comp_end, fin;

    float* res;
    laik_map_def1(result, (void**) &res, 0);

    // Loop over all local slices
    for(int sNo = 0; ; sNo++) {
        //Laik_Slice* slc = laik_my_slice(p, sNo);
        Slice slice;
        Laik_TaskSlice* slc = laik_my_slice_1d(p, sNo, &slice.from, &slice.to);
        

        if (slc == 0) break;
        
        laik_log(LAIK_LL_Debug, "My Slice: from %lu, to %lu", slice.from, slice.to);

        //auto fromRow = slc->from.i[0];
        //auto toRow = slc->to.i[0];
        //matrix.mapRows(fromRow, toRow - fromRow);

        RowData rv = cache_get(matrix, slice);
        auto fromRow = rv.fromRow;
        auto toRow = rv.toRow;
        matrix.mapRows(fromRow, toRow - fromRow + 1);
        laik_log(LAIK_LL_Debug, "My True Slice: from %lu, to %lu", fromRow, toRow);

        s_comp = std::chrono::system_clock::now();

        for(size_t r = 0; r < rv.rowIdx.size() - 1; r++) {
            std::for_each(matrix.beginRow(r, rv.rowIdx), matrix.endRow(r, rv.rowIdx),
                [&](const RowElement<float>& e){ res[r + fromRow] += (float)e.value() * input[e.column()]; });
            
            if (r<100 ) laik_log(LAIK_LL_Debug, "index: %lu, value:%f", r+fromRow, res[r+fromRow]);
        }

        s_comp_end = std::chrono::system_clock::now();
        comp_time += (s_comp_end - s_comp);
    }

    //laik_switchto_flow(result, LAIK_DF_CopyIn);
    laik_switchto_flow(result, LAIK_DF_Preserve, LAIK_RO_Sum);
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


void calcBkProj(Laik_Partitioning* p,
                const Csr4Matrix& matrix,
                const Vector<float>& correlation,
                Laik_Data* update,
                std::chrono::duration<float>& comp_time,
                std::chrono::duration<float>& total_time)
{
    //laik_switchto_flow(update, LAIK_DF_Init | LAIK_DF_ReduceOut | LAIK_DF_Sum);
    laik_switchto_flow(update, LAIK_DF_Init, LAIK_RO_Sum);

    float* res;
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> s_comp, s_comp_end,fin;

    laik_map_def1(update, (void**) &res, 0);

    // Loop over all local slices
    for(int sNo = 0; ; sNo++) {
        //Laik_Slice* slc = laik_my_slice(p, sNo);

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


    //laik_switchto_flow(update, LAIK_DF_CopyIn);
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


void mlem(Laik_Instance* inst, Laik_Group* world, Laik_Partitioning* part,
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

    Vector<float> correlation(nRows, 0.0);

    Laik_Data* update = laik_new_data_1d(inst, laik_Float, nColumns);
    laik_data_set_name(update, "update");
    laik_switchto_new_partitioning(update, world, laik_All, LAIK_DF_None, LAIK_RO_None);

    // Calculate column sums ("norm")
    Laik_Data* norm = laik_new_data_1d(inst, laik_Float, nColumns);
    laik_data_set_name(norm, "norm");
    laik_switchto_new_partitioning(norm, world, laik_All, LAIK_DF_None, LAIK_RO_None);

    // Set "all" partitionings for data containers: only for reductions
    // Avoids creation of new partitionings as switching flows is enough
    // laik_switchto_new(fwproj, laik_All, LAIK_DF_None);
    // laik_switchto_new(update, laik_All, LAIK_DF_None);
    // laik_switchto_new(norm, laik_All, LAIK_DF_None);

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    calcColumnSums(part, matrix, norm);
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
#ifdef MESSUNG
    int flag;
#endif
    for (int iter=0; iter<nIterations; ++iter) {

        compute_time = std::chrono::duration<float>::zero();
        total_time = std::chrono::duration<float>::zero();
        
        calcFwProj(part, matrix, image, fwproj, compute_time, total_time);
        calcCorrel(fwproj, lmsino, correlation, compute_time, total_time);
        calcBkProj(part, matrix, correlation, update, compute_time, total_time);
        calcUpdate(update, norm, image, compute_time, total_time);


#ifndef MESSUNG
        // debug: sum over image values
        float s = 0.0;
        for(uint i = 0; i < matrix.columns(); i++) s += image[i];
        laik_log(LAIK_LL_Info, "Finished Iteration %d, Time: %f(%f/%f/%f)\n",
                 iter + 1,
                 total_time.count(),
                 compute_time.count(),
                 laik_get_total_time() - laik_get_backend_time(),
                 laik_get_backend_time()
                 );
#else
        printf("%d, %d, %lf, %lf, %lf, %lf\n",
               flag,
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
        // Static repartitioning
        start = std::chrono::system_clock::now();
        char* shrink_iter_str = getenv("SHRINK_ITER");
        int shrink_iter = shrink_iter_str ? atoi(shrink_iter_str) : 0;
        //Laik_Group* g = laik_get_pgroup(part);
        int gsize = laik_size(world);
        if ((shrink_iter > 0) && (iter == shrink_iter) && (gsize > 1)) {
            char s[500];
            int removeLen = 0;
            int removeList[gsize];
            int o = 0;
            char* from_str = getenv("SHRINK_FROM");
            int from = from_str ? atoi(from_str) : -1;
            char* to_str   = getenv("SHRINK_TO");
            int to = to_str ? atoi(to_str) : -1;
            char* repart_str = getenv("REPART_TYPE");
            int repart_type = repart_str ? atoi(repart_str) : -1;
            if ((from == -1) && (to == -1)) {
                // Remove half the tasks (all odd)
                for(int task = 1; task < gsize; task += 2) {
                    removeList[removeLen++] = task;
                    o += sprintf(s+o, "%d ", task);
                }
            }
            else {
                for(int task = 0; task < gsize; task++) {
                    if ((from == -1) || (task >= from)) {
                        if ((to == -1) || (task <= to)) {
                            removeList[removeLen++] = task;
                            o += sprintf(s+o, "%d ", task);
                        }
                    }
                }
            }
            laik_log(LAIK_LL_Info, "Iteration %d: Removing ( %s), size %d => %d",
                     iter, s, gsize, gsize - removeLen);
            if (removeLen == gsize)
                laik_panic("No task left!");

            Laik_Group* g2;
            g2 = laik_new_shrinked_group(world, removeLen, removeList);

            //Laik_Partitioner* rep = 0;
            Laik_Partitioning* p2;
            Laik_Space* space = laik_partitioning_get_space(part);
            laik_log(LAIK_LL_Info, "BLOB------\n");
             Laik_Partitioner* part2;
            switch(repart_type){
                case 1:
                {
                    //rep = laik_new_reassign_partitioner(g2, getEW, (void*)&matrix);

                    part2 = laik_new_reassign_partitioner(g2, 0, 0);
                    p2 = laik_new_partitioning(part2, world, space, part);
                    laik_partitioning_migrate(p2, g2);
                    break;
                }
                default:
                {
                    part2 = laik_new_block_partitioner1();
                    p2 = laik_new_partitioning(part2, g2, space, 0);
                    break;
                }
            }
            laik_log(LAIK_LL_Info, "Repart Type %d", repart_type);
            part = p2;
            world = g2; 
            
            //laik_migrate_and_repartition(part, g2, rep);

            // For the All partitionings, nothing is repartitioned,
            // but data is only preserved for Copy/ReduceOut flows (only norm)
            //laik_migrate_and_repartition(laik_get_active(fwproj), g2, 0);
            Laik_Partitioning* pfwprojAll2 = laik_new_partitioning(laik_All, g2,
                            laik_data_get_space(fwproj), 0);
            laik_log(LAIK_LL_Info, "fwproj------\n");
            laik_switchto_partitioning(fwproj, pfwprojAll2, LAIK_DF_None, LAIK_RO_None);
            //laik_migrate_and_repartition(laik_get_active(update), g2, 0);
            Laik_Partitioning* pupdateAll2 = laik_new_partitioning(laik_All, g2,
                            laik_data_get_space(update), 0);
            laik_switchto_partitioning(update, pupdateAll2, LAIK_DF_None, LAIK_RO_None);
            //laik_migrate_and_repartition(laik_get_active(norm), g2, 0);
            Laik_Partitioning* pnormAll2 = laik_new_partitioning(laik_All, g2,
                            laik_data_get_space(norm), 0);
            laik_switchto_partitioning(norm, pnormAll2, LAIK_DF_None, LAIK_RO_None);
            end = std::chrono::system_clock::now();
            elapsed_seconds = end - start;
#ifdef MESSUNG
            flag = 1;
#endif
#ifdef MESSUNG
            printf("%d, %d, %lf, %lf, %lf, %lf\n",
                   -255,
                   -255,
                   elapsed_seconds.count(),
                   laik_get_total_time(),
                   laik_get_total_time() - laik_get_backend_time(),
                   laik_get_backend_time()
                   );
            laik_reset_profiling(inst);
#endif
            if (laik_myid(g2) == -1){
            laik_log(LAIK_LL_Debug, "myid is -1, exiting =====");
            break;
             }
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
#ifndef MESSUNG
    std::cout << "Matrix rows (LORs): " << matrix.rows() << std::endl;
    std::cout << "Matrix cols (VOXs): " << matrix.columns() << std::endl;
#endif
    Vector<int> lmsino(progops.infilename);
    Vector<float> image(matrix.columns(), 0.0);

    Laik_Instance* inst = laik_init_mpi(&argc, &argv);
    Laik_Group* world = laik_world(inst);

    laik_enable_profiling(inst);
    
    // Laik_Space* space;
    // Laik_Partitioner* part;
    // Laik_Partitioning* p;
    // space = laik_new_space_1d(inst, matrix.rows());
    // part = laik_new_block_partitioner_iw1(getEW, &matrix);
    // p = laik_new_partitioning(world, space, part);
    //auto ranges = partition(laik_size(world), matrix);

    // 1d space, block partitioning of matrix elements
    Laik_Space* space = laik_new_space_1d(inst, matrix.elements());
    Laik_Partitioner* part = laik_new_block_partitioner1();
    Laik_Partitioning* p = laik_new_partitioning(part, world, space, 0);

    mlem(inst, world, p, matrix, lmsino, image, progops.iterations);

    if (laik_myid(world) == 0)
        image.writeToFile(progops.outfilename);

    laik_finalize(inst);

    return 0;
}
