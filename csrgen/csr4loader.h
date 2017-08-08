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


#ifndef CSR4LOADER_H
#define CSR4LOADER_H

#include "csr4.h"
#include "csr4errors.h"
#include "fileIO.h"

namespace CSR {
  template<typename T>
  class CSR4Loader : FileIO {
    public:
      CSR4Loader();
      ~CSR4Loader();

      virtual CsrResult open(const std::string& filename);
      virtual CsrResult close();

      CsrResult load(CSR::CSR4<T>& CSR4) const;

    private:
      CsrResult loadHeader(CSR4Header& header, uint32_t bytesize) const;
      CsrResult loadConfig(ScannerConfig& config) const;
      CsrResult loadIndices(std::vector<uint64_t>& rowIndices) const;
      CsrResult loadData(std::vector< std::pair<uint32_t, T> >& data) const;

      mutable std::ifstream ifs;
  };
}

#endif
