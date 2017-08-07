/**
  csr4gen

  Copyright © 2017 Thorsten Fuchs

  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/**
 * @file csr4storer.cc
 * @author Thorsten Fuchs
 * @date 4 Jul 2017
*/

#include <stdio.h>

#include "csr4storer.h"

using namespace CSR;

template<typename T>
CSR4Storer<T>::CSR4Storer() {

}

template<typename T>
CSR4Storer<T>::~CSR4Storer() {

}

template<typename T>
CsrResult CSR4Storer<T>::save(const CSR4<T>& csr) const {
  assert(ofs.is_open());

  writeHeader(csr.getHeader(), csr.getHeaderLength());
  writeConfig(csr.getScanConfig());
  writeIndices(csr.getRowIndices());
  writeData(csr.getData());

  return CSR_SUCCESS;
}

template <typename T>
CsrResult CSR4Storer<T>::open(const std::string& filename) {
  ofs.open(filename, std::ios::out|std::ios::binary|std::ios::trunc);
  if(!ofs.is_open()) {
    std::cerr <<
      "Error opening file " << filename << " for writing!" << std::endl;
    return CSR_IO_OPEN_ERR;
  }
  return CSR_SUCCESS;
}

template <typename T>
CsrResult CSR4Storer<T>::close() {
  ofs.close();
  if(ofs.fail()) {
    std::cerr << "Error closing file!" << std::endl;
    return CSR_IO_CLOSE_ERR;
  }
  return CSR_SUCCESS;
}

template <typename T>
CsrResult CSR4Storer<T>::writeHeader(
    const CSR::CSR4Header& header
    , uint32_t bytesize
) const {
  ofs.write(reinterpret_cast<const char*>(&header), bytesize);

  return CSR_SUCCESS;
}

template <typename T>
CsrResult CSR4Storer<T>::writeConfig(const ScannerConfig& config) const {
  const char* pvar; // Pointer to (casted) data
  std::streamsize bytesize; // Size of data

  // nLayers
  writeU32(config.nLayers());

  // layers
  bytesize = config.nLayers() * 5 * sizeof(float);
  pvar = reinterpret_cast<const char*>(config.getLayers().data());
  ofs.write(pvar , bytesize);

  // nBlocks
  writeU32(config.nBlocks());

  // blocksize
  writeU32(config.blocksize());

  // block gaps
  bytesize = config.nBlockGaps() * sizeof(float);
  pvar = reinterpret_cast<const char*>(config.getBlockGaps().data());
  ofs.write(pvar, bytesize);

  // nRings
  writeU32(config.nRings());

  // ring gaps
  bytesize = config.nRingGaps() * sizeof(float);
  pvar = reinterpret_cast<const char*>(config.getRingGaps().data());
  ofs.write(pvar, bytesize);

  // voxel size
  bytesize = 3 * sizeof(float);
  pvar = reinterpret_cast<const char*>(&config.getVoxelSize());
  ofs.write(pvar, bytesize);

  // grid dimensions
  bytesize = 3 * sizeof(uint32_t);
  pvar = reinterpret_cast<const char*>(&config.getGridDim());
  ofs.write(pvar, bytesize);

  return CSR_SUCCESS;
}

template <typename T>
CsrResult CSR4Storer<T>::writeIndices(
    const std::vector<uint64_t>& rowIndices
) const {
  std::streamsize bytesize = rowIndices.size() * sizeof(uint64_t);
  ofs.write(reinterpret_cast<const char*>(rowIndices.data()), bytesize);

  return CSR_SUCCESS;
}

// TODO: What if pair has double?
template <typename T>
CsrResult CSR4Storer<T>::writeData(
    const std::vector< std::pair<uint32_t,T> >& data
) const {
  std::streamsize bytesize = data.size() * (sizeof(uint32_t) + sizeof(T));
  ofs.write(reinterpret_cast<const char*>(data.data()), bytesize);

  return CSR_SUCCESS;
}

template <typename T>
CsrResult CSR4Storer<T>::writeU32(uint32_t var) const {
  ofs.write(reinterpret_cast<const char*>(&var), sizeof(uint32_t));

  return CSR_SUCCESS;
}

// Instantiate templated function for all supported types
template class CSR::CSR4Storer<float>;
