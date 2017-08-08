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

#ifndef CSR4GENERATORRAND_H
#define CSR4GENERATORRAND_H

#include "csr4generator.h"

namespace CSR {
  template<typename T>
  class CSR4GeneratorRand : CSR4Generator<T> {
    public:
      CSR4GeneratorRand();
      ~CSR4GeneratorRand();

      /// Fills rowIndices and data with fitting random values
      /**
       * \param[out] csrSR4 matrix, must contain valid scanner config
       * \param[in] config  void Filename of a scanner config
       * \param[in] density Density of matrix
       * \param[in] variance Variance in produced rows around mean
       */
      virtual CsrResult generate(
          CSR4<T>& csr
        , const std::string& config
        , float density
        , float variance
      );

    private:
      CsrResult generateRows(CSR4<T>& csr, float density, float variance);
      CsrResult generateData(CSR4<T>& csr, T randFunc());
  };
}

#endif
