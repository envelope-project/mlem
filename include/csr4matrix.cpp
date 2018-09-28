/**
        csr4matrix.cpp

        Created on: Oct 15, 2009
                Author: kuestner
*/
#define O_LARGEFILE 0

#include <string>
using std::string;
#include <sstream>
using std::stringstream;
#include <stdexcept>
using std::runtime_error;

#include <stdint.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <assert.h>
#include <cstring>

#ifdef _HPC_
    #include <stdlib.h>
    #include <string.h>
#endif

#ifdef XEON_PHI
    #include <hbwmalloc.h>
#endif

#include "csr4matrix.hpp"

Csr4Matrix::Csr4Matrix(const string& fn)
{
    char* map;
    filename = fn;

    // Cannot use the C++ iostream library, as it does not provide a way
    // to access the underlying file descriptor
    fileDescriptor = open(filename.c_str(), O_RDONLY);
    if (fileDescriptor < 0)
        throw runtime_error(string("Cannot open file ") + filename);

    // Alternative for seek would be getting the file size via stat
    fileSize = lseek64(fileDescriptor, 0, SEEK_END);
    lseek64(fileDescriptor, 0, SEEK_SET);

    if (fileSize < minHeaderLength)
        throw runtime_error(string("File too short ") + filename);

    // Make this big enough for header + scanConfig
    int headerLength = 1000;
    if (fileSize < headerLength) headerLength = fileSize;

    #ifdef XEON_PHI
        map = (char*) hbw_malloc(headerLength);
    #else 
        map = (char*) malloc(headerLength);
    #endif

    if (map == 0)
        throw runtime_error(string("Cannot alloc header "));
    if (read(fileDescriptor, map, headerLength) < headerLength)
        throw runtime_error(string("Cannot read header ") + filename);

    string magic = string((char*)map, 4);
    uint8_t version = *((uint8_t*)map + 4);

    flags = *((uint8_t*)(map + 5));
    if (flags & xsymmask) symcfg.x = true;
    if (flags & ysymmask) symcfg.y = true;
    if (flags & zsymmask) symcfg.z = true;

    nRows = *((uint32_t*)(map + 8));
    nColumns = *((uint32_t*)(map + 12));
    nnz = *((uint64_t*)(map + 16));

    scanConfigBytes = *((uint32_t*)(map + 24));

    if (magic != string("PCSR")) {
        stringstream ss;
        ss<<"Input file '"<<filename<<"' is not a CSR matrix";
        throw runtime_error(ss.str());
    }

    if (version != 4) {
        stringstream ss;
        ss<<"Input file '"<<filename<<"' has wrong CSR version (need v4)";
        throw runtime_error(ss.str());
    }

    uint32_t size = minHeaderLength + scanConfigBytes
            + nRows * sizeof(uint64_t) + nnz * sizeof(RowElement<float>);
    if (fileSize < size) {
        stringstream ss;
        ss<<"Input file '"<<filename<<"' is too short: "<<fileSize<<" B"
         <<" instead of at least "<<size<<" B";
        throw runtime_error(ss.str());
    }

    if (scanConfigBytes < minScanConfigBytes) {
        throw runtime_error(string(
                                "Cannot find scanner geometry configuration"));
    }

    // Start reading scanner geometry configuration
    char* p = map + minHeaderLength; // p helper pointer
    uint32_t nLayers = *((uint32_t*)p); p += sizeof(uint32_t);
    for (uint32_t i=0; i<nLayers; ++i) {
        float w  = *((float*)p); p+= sizeof(float);
        float h  = *((float*)p); p+= sizeof(float);
        float d  = *((float*)p); p+= sizeof(float);
        float r  = *((float*)p); p+= sizeof(float);
        float mu = *((float*)p); p+= sizeof(float);
        scancfg.addLayer(LayerConfig(w, h, d, r, mu));
    }
    uint32_t nBlocks = *((uint32_t*)p); p += sizeof(uint32_t);
    scancfg.setNBlocks(nBlocks);
    uint32_t blocksize = *((uint32_t*)p); p += sizeof(uint32_t);
    for (uint32_t i=0; i<blocksize-1; ++i) { // nr of gaps is one less
        float gap =  *((float*)p); p+= sizeof(float);
        scancfg.addBlockGap(gap);
    }
    uint32_t nRings = *((uint32_t*)p); p += sizeof(uint32_t);
    for (uint32_t i=0; i<nRings-1; ++i) { // nr of gaps is one less
        float gap =  *((float*)p); p+= sizeof(float);
        scancfg.addRingGap(gap);
    }
    float w = *((float*)p); p+= sizeof(float);
    float h = *((float*)p); p+= sizeof(float);
    float d = *((float*)p); p+= sizeof(float);
    scancfg.setVoxelSize(Vector3d<float>(w, h, d));
    uint32_t nx = *((uint32_t*)p); p += sizeof(uint32_t);
    uint32_t ny = *((uint32_t*)p); p += sizeof(uint32_t);
    uint32_t nz = *((uint32_t*)p); p += sizeof(uint32_t);
    scancfg.setGridDim(IVec3(nx, ny, nz));

    if (p - (map + minHeaderLength) != scanConfigBytes) {
        throw runtime_error(string(
                                "Error reading scanner geometry configuration"));
    }
    if (p - map > headerLength) {
        throw runtime_error(string(
                                "Internal Error: headerLength too small"));
    }

    #ifdef XEON_PHI
        hbw_free(map);
    #else 
        free(map); 
    #endif

    // Read row index
    ssize_t rowidxLength = (ssize_t) nRows * sizeof(uint64_t);
    #ifdef XEON_PHI
        rowidx = (uint64_t*) hbw_malloc(rowidxLength); 
    #else 
        rowidx = (uint64_t*) malloc(rowidxLength); 
    #endif

    if (rowidx == 0)
        throw runtime_error(string("Cannot alloc row indexes"));
    lseek(fileDescriptor, minHeaderLength + scanConfigBytes, SEEK_SET);
    if (read(fileDescriptor, rowidx, rowidxLength) < rowidxLength)
        throw runtime_error(string("Cannot read row indexes ") + filename);

    // Nothing mapped yet
    memset(mappings, 0x0, sizeof(mappings));
    memset(&activeMappings, 0x0, sizeof(int));
    currentMap = -1;
    pageSize = sysconf(_SC_PAGE_SIZE);

    // Piecewise mapping on demand, else map all now:
    // mapRows(0, nRows);
}

Csr4Matrix::~Csr4Matrix()
{
    mapRows(0,0); // unmap

    #ifdef XEON_PHI
        hbw_free(rowidx); 
    #else 
        free(rowidx); 
    #endif

    #ifdef _HPC_
        for(size_t i = 0; i < this->activeMappings; i++) {
            struct tag_mappings *cur = &(this->mappings[i]);
            free(cur->map2); 
        }
    #endif

    close(fileDescriptor);
}

void Csr4Matrix::mapRows(uint32_t start, uint32_t count) const
{
    uint64_t mapData, mapStart, mapEnd;

    for(int i=0; i<this->activeMappings; i++){
        if ((this->mappings[i].mapRowStart == start) && (this->mappings[i].mapRowCount == count)){
            this->currentMap = i;
            #ifdef _HPC_
                data = (RowElement<float>*) (this->mappings[i].map2 + this->mappings[i].off);
            #else
                data = (RowElement<float>*) (this->mappings[i].map + this->mappings[i].off);
            #endif
            return;
        }
    }

    // FIXME@JW: LRU/Unmap for overlapping

    if (count == 0) return;
    if (start >= nRows) throw runtime_error(string("Wrong row map request"));
    if (start + count > nRows) count = nRows - start;

    // Start offset of data array in file
    mapData = minHeaderLength + scanConfigBytes + nRows * sizeof(uint64_t);
    mapStart = mapData;
    if (start > 0) mapStart += rowidx[start -1] * sizeof(RowElement<float>);
    mapEnd = mapData + rowidx[start + count -1] * sizeof(RowElement<float>);
    assert(this->activeMappings < MAX_MAPPINGS);

    struct tag_mappings *cur = &(this->mappings[this->activeMappings]);

    cur->mapSize = mapEnd - mapStart;
    cur->mapRowStart = start;
    cur->mapRowCount = count;
    cur->mapFirstElement = (start > 0) ? rowidx[start -1] : 0;

    // Start map at multiples of pages size
    cur->off = mapStart & (pageSize - 1);
    mapStart -= cur->off;
    cur->mapSize += cur->off;

    cur->map = (char*)mmap(0, cur->mapSize, PROT_READ, MAP_PRIVATE, fileDescriptor, mapStart);

    if (cur->map == MAP_FAILED)
        throw runtime_error(string("Cannot map file ") + filename);

    #ifdef _HPC_        
        memcpy(cur->map2, cur->map, cur->mapSize);
        data = (RowElement<float>*) (cur->map2 + cur->off);
    #else
        data = (RowElement<float>*) (cur->map + cur->off);
    #endif
    this->currentMap = this->activeMappings;
    this->activeMappings ++;
}

#if 0
void Csr4Matrix::mapRow(uint32_t row) const
{
    // already mapped?
    if ((row >= mapRowStart) && (row < mapRowStart + mapRowCount)) return;

    // map 10000 lines starting from row
    mapRows(row, 10000);
}
#endif

uint32_t Csr4Matrix::elementsInRow(uint32_t rowNr) const
{
    if (rowNr == 0) return rowidx[rowNr];
    else return rowidx[rowNr] - rowidx[rowNr - 1];
}

Csr4Matrix::RowIterator Csr4Matrix::beginRow(uint32_t rowNr) const {
    if (rowNr == 0) return RowIterator(data);
    else return RowIterator(&data[rowidx[rowNr - 1]]);
}

Csr4Matrix::RowIterator Csr4Matrix::endRow(uint32_t rowNr) const {
    return RowIterator(&data[rowidx[rowNr]]);
}
