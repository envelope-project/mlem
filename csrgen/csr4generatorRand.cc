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

#include <random>
#include <omp.h>

#include "csr4generatorRand.h"
#include "fisher.h"
#include "configloader.h"

using namespace CSR;

template<typename T>
CSR4GeneratorRand<T>::CSR4GeneratorRand() {

}

template<typename T>
CSR4GeneratorRand<T>::~CSR4GeneratorRand() {

}

template<typename T>
CsrResult CSR4GeneratorRand<T>::generate(
    CSR4<T>& csr
  , const std::string& config
  , float density
  , float variance
) {
  CsrResult res;

  ConfigLoader confloader(config);
  ScannerConfig scanConfig = confloader.getConfig();

  // Check config variables
  assert(scanConfig.nVoxels() != 0);
  assert(scanConfig.nLors() != 0);
  assert(density > 0.f && density <= 1.f);

  csr.reset();
  csr.setScanConfig(scanConfig);
  csr.setNumRows(scanConfig.nLors());
  csr.setNumCols(scanConfig.nVoxels());

  std::cerr << "Generating rows!" << std::endl;
  res = generateRows(csr, density, variance);
  if(res != CSR_SUCCESS) return res;
  std::cerr << "Generating cols!" << std::endl;
  // TODO: Better generator for data values
  res = generateData(csr, [](){ return static_cast<T>(1.0); });
  if(res != CSR_SUCCESS) return res;

  csr.setPopulated(true);

  return res;
}

template<typename T>
CsrResult CSR4GeneratorRand<T>::generateRows(
    CSR4<T>& csr
    , float density
    , float variance
) {
  CsrResult ret{CSR_SUCCESS};
  uint64_t acc;
  uint64_t reqElemCount;
  uint32_t avgRowElemCount;
  double tmp;
  std::default_random_engine rng;

  acc = {0};
  reqElemCount = csr.getNumRows() * csr.getNumCols() * density;
  avgRowElemCount = reqElemCount / csr.getNumRows();

  std::normal_distribution<double> normdist(avgRowElemCount, variance);

  // Resize rowIndices to fit new values
  ret = csr.resizeRowIndices();
  if(ret != CSR_SUCCESS) {
    return ret;
  }

  // Generate a normal distributed length for each row
  for(uint32_t i=0; i<csr.getNumRows(); ++i) {
    // Clamp tmp to valid values
    tmp = normdist(rng);
    if(tmp < 0.0) tmp = 0.0;
    else if(tmp > csr.getNumCols()) tmp = csr.getNumCols();

    acc += static_cast<uint64_t>(tmp);
    csr.setRowIndex(i, acc);
  }

  csr.setNNZ(acc);

  return ret;
}

template<typename T>
CsrResult CSR4GeneratorRand<T>::generateData(
    CSR4<T>& csr,
    T randFunc()
) {
  CsrResult ret{CSR_SUCCESS};
  uint32_t rowElemCount{0};
  uint32_t i,j;
  std::vector<uint32_t> shufflePad;

  shufflePad.resize(csr.getNumCols());

  // Resize data to fit new values
  ret = csr.resizeData();
  if(ret != CSR_SUCCESS) {
    return ret;
  }

  uint32_t processedRows{0};
  uint32_t t{0};
  rowElemCount = csr.getRowIndex(0);
  fisher_shuffle(shufflePad.data(), csr.getNumCols(), rowElemCount);
  for(j=0; j<rowElemCount; ++j) {
    T value = randFunc();
    csr.setDatum(j, std::pair<uint32_t,T>{shufflePad[j], value});
  }

#pragma omp parallel private(j, rowElemCount)
  {
    // Create local shufflePad for OpenMP threads
    std::vector<uint32_t> shufflePadLocal;
    shufflePadLocal.resize(csr.getNumCols());

#pragma omp for schedule(guided)
    for(i=1; i<csr.getNumRows(); ++i) {
      // Print status of current generation
#ifdef USE_OPENMP
      if(omp_get_thread_num() == 0)
#endif
      {
        if((static_cast<float>(processedRows) / (csr.getNumRows()-1) * 100.f) >= t) {
          std::cerr << t << "%" << std::endl;
          t += 1;
        }
      }

#pragma omp atomic
      processedRows++;

      // Calculate number of row elements and fill them
      rowElemCount = csr.getRowIndex(i) - csr.getRowIndex(i-1);
      if(rowElemCount == 0) continue;
      fisher_shuffle(shufflePadLocal.data(), csr.getNumCols(), rowElemCount);
      for(j=0; j<rowElemCount; ++j) {
        T value = randFunc();
        csr.setDatum(csr.getRowIndex(i-1)+j, std::pair<uint32_t,T>{shufflePadLocal[j], value});
      }
    }
  }
  std::cout << "Finished generating!" << std::endl;

  return ret;
}

template class CSR::CSR4GeneratorRand<float>;
