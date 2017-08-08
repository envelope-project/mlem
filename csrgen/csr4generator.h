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


#ifndef CSR4GENERATOR_H
#define CSR4GENERATOR_H

#include "csr4.h"
#include "csr4errors.h"

namespace CSR {
  /// Interface class for generators
  template<typename T>
  class CSR4Generator {
    private:
      /// Virtual function to generate CSR4 data
      /**
       * \param[in,out] csr    CSR4 matrix, must contain valid scanner config
       * \param[in]     config void pointer to generator specific config struct
       */
      //virtual CsrResult generate(CSR4<T>& csr, svoid* config) = 0;
  };
}

#endif
