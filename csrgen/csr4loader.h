/**
  csr4gen

  Copyright © 2017 Thorsten Fuchs

  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/**
 * @file csr4loader.h
 * @author Thorsten Fuchs
 * @date 4 Jul 2017
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
