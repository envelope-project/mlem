/**
    Copyright © 2017 Thorsten Fuchs
    Copyright © 2017 LRR, TU Muenchen
    Authors: Thorsten Fuchs

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


#include <stdio.h>

#include "csr4loader.h"

using namespace CSR;

template<typename T>
CSR4Loader<T>::CSR4Loader() {

}

template<typename T>
CSR4Loader<T>::~CSR4Loader() {

}

template<typename T>
CsrResult CSR4Loader<T>::open(const std::string& filename) {
  ifs.open(filename, std::ios::in|std::ios::binary);
  if(!ifs.is_open()) {
    std::cerr <<
      "Error opening file " << filename << "for reading!" << std::endl;
    return CSR_IO_OPEN_ERR;
  }
  return CSR_SUCCESS;
}

template<typename T>
CsrResult CSR4Loader<T>::close() {
  ifs.close();
  if(ifs.fail()) {
    std::cerr << "Error closing file!" << std::endl;
    return CSR_IO_CLOSE_ERR;
  }
  return CSR_SUCCESS;
}

template<typename T>
CsrResult CSR4Loader<T>::load(CSR4<T>& csr) const {
  assert(ifs.is_open());

  CsrResult res;

  CSR4Header header;
  ScannerConfig config;
  std::vector< uint64_t > rowIndices;
  std::vector< std::pair< uint32_t,T > > data;

  res = loadHeader(header, csr.getHeaderLength());
  if(res != CSR_SUCCESS) return res;
  csr.setHeader(header);

  res = loadConfig(config);
  if(res != CSR_SUCCESS) return res;
  csr.setScanConfig(config);

  rowIndices.resize(csr.getNumRows());
  res = loadIndices(rowIndices);
  if(res != CSR_SUCCESS) return res;
  csr.setRowIndices(std::move(rowIndices));

  data.resize(csr.getNNZ());
  res = loadData(data);
  if(res != CSR_SUCCESS) return res;
  csr.setData(std::move(data));

  csr.setPopulated(true);

  return CSR_SUCCESS;
}

template<typename T>
CsrResult CSR4Loader<T>::loadHeader(
    CSR4Header& header
    , uint32_t bytesize
) const {
  // Read complete header and check if filestream still good
  ifs.read(reinterpret_cast<char*>(&header), bytesize);
  if(!ifs.good()) {
    std::cerr << "Error reading header!" << std::endl;
    return CSR_IO_LOAD_ERR;
  }

  // Check header fields
  if(strncmp(header.magic, "PCSR", 4) != 0) {
    std::cerr << "Magic number wrong!" << std::endl;
    return CSR_IO_LOAD_ERR;
  }

  if(header.version != 4) {
    std::cerr << "Wrong version!" << std::endl;
    return CSR_IO_LOAD_ERR;
  }

  if(header.scanConfigBytes < CSR4<T>::minScanBytes) {
    std::cerr << "Scanner config size to small!" << std::endl;
    return CSR_IO_LOAD_ERR;
  }

  return CSR_SUCCESS;
}

template<typename T>
CsrResult CSR4Loader<T>::loadConfig(ScannerConfig& config) const {
  // Read layers
  uint32_t nLayers;
  ifs.read(reinterpret_cast<char*>(&nLayers), sizeof(uint32_t));
  for(uint32_t i=0; i<nLayers; ++i) {
    LayerConfig layer;
    ifs.read(reinterpret_cast<char*>(&layer), sizeof(float)*5);
    config.addLayer(layer);
  }
  // Read block number
  uint32_t nBlocks;
  ifs.read(reinterpret_cast<char*>(&nBlocks), sizeof(uint32_t));
  config.setNBlocks(nBlocks);
  // Read block gaps
  uint32_t blocksize;
  ifs.read(reinterpret_cast<char*>(&blocksize), sizeof(uint32_t));
  for(uint32_t i=0; i<blocksize-1; ++i) {
    float gap;
    ifs.read(reinterpret_cast<char*>(&gap), sizeof(float));
    config.addBlockGap(gap);
  }
  // Read ring gaps
  uint32_t nRings;
  ifs.read(reinterpret_cast<char*>(&nRings), sizeof(uint32_t));
  for(uint32_t i=0; i<nRings-1; ++i) {
    float gap;
    ifs.read(reinterpret_cast<char*>(&gap), sizeof(float));
    config.addRingGap(gap);
  }
  // Read voxel size
  Vector3D<float> vsize;
  ifs.read(reinterpret_cast<char*>(&vsize), sizeof(float)*3);
  config.setVoxelSize(vsize);
  Vector3D<uint32_t> dims;
  ifs.read(reinterpret_cast<char*>(&dims), sizeof(uint32_t)*3);
  config.setGridDim(dims);

  return CSR_SUCCESS;
}

template<typename T>
CsrResult CSR4Loader<T>::loadIndices(std::vector<uint64_t>& rowIndices) const {
  ifs.read(reinterpret_cast<char*>(rowIndices.data()), rowIndices.size()*sizeof(uint64_t));

  return CSR_SUCCESS;
}

template<typename T>
CsrResult CSR4Loader<T>::loadData(std::vector< std::pair<uint32_t, T> >& data) const {
  ifs.read(reinterpret_cast<char*>(data.data()), data.size()*sizeof(std::pair<uint32_t,T>));

  return CSR_SUCCESS;
}

template class CSR::CSR4Loader<float>;
